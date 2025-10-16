from __future__ import annotations as _annotations

from typing import Annotated
from pydantic_graph import Edge
from dataclasses import dataclass, field
from pydantic import BaseModel
from pydantic_graph import (
    BaseNode,
    End,
    Graph,
    GraphRunContext,
)
from pydantic_ai import Agent, format_as_xml
from pydantic_ai import ModelMessage
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic_graph.persistence.file import FileStatePersistence
import uuid

# Load environment variables from .env file
load_dotenv("./.env", override=True)

class LLMSettings(BaseSettings):
    GEMINI_API_KEY: str
    GEMINI_MODEL_NAME: str
    TEMPERATURE: float

class ClinicalLLMAgent(Agent):
    def __init__(self, settings: LLMSettings, result_type: BaseModel):
        provider = GoogleProvider(
            api_key=settings.GEMINI_API_KEY
        )
        model_settings = GoogleModelSettings(
            temperature=settings.TEMPERATURE
        )
        model = GoogleModel(
            model_name=settings.GEMINI_MODEL_NAME,
            provider=provider,
        )

        super().__init__(model=model, model_settings=model_settings, output_type=result_type)

ask_agent = ClinicalLLMAgent(
    LLMSettings(), str
)


@dataclass
class QuestionState:
    question: str | None = None
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class Ask(BaseNode[QuestionState]):
    """Generate question using GPT-4o."""
    docstring_notes = True
    async def run(
        self, ctx: GraphRunContext[QuestionState]
    ) -> Annotated[Answer, Edge(label='Ask the question')]:
        result = await ask_agent.run(
            'Ask a simple question with a single correct answer.',
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += result.new_messages()
        ctx.state.question = result.output
        return Answer(result.output)


@dataclass
class Answer(BaseNode[QuestionState]):
    question: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
        # This will be handled by the API endpoint
        pass


class EvaluationResult(BaseModel, use_attribute_docstrings=True):
    correct: bool
    """Whether the answer is correct."""
    comment: str
    """Comment on the answer, reprimand the user if the answer is wrong."""


evaluate_agent = ClinicalLLMAgent(
    LLMSettings(), EvaluationResult
)

@evaluate_agent.system_prompt
def system_prompt():
    return (
        "You are a strict and concise teacher. Evaluate the student's answer to the question. "
        "If the answer is correct, respond with {'correct': true, 'comment': 'Well done!'}. "
        "If the answer is incorrect, respond with {'correct': false, 'comment': 'The correct answer is ...'}. "
        "Always respond in JSON format."
    )

@dataclass
class Evaluate(BaseNode[QuestionState, None, str]):
    answer: str

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> Annotated[End[str], Edge(label='success')] | Reprimand:
        assert ctx.state.question is not None
        result = await evaluate_agent.run(
            format_as_xml({'question': ctx.state.question, 'answer': self.answer}),
            message_history=ctx.state.evaluate_agent_messages,
        )
        ctx.state.evaluate_agent_messages += result.new_messages()
        if result.output.correct:
            return End(result.output.comment)
        else:
            return Reprimand(result.output.comment)


@dataclass
class Reprimand(BaseNode[QuestionState]):
    comment: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Ask:
        ctx.state.question = None
        return Ask()


question_graph = Graph(
    nodes=(Ask, Answer, Evaluate, Reprimand), state_type=QuestionState
)

# FastAPI Application
app = FastAPI(title="Question-Answer API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QuestionResponse(BaseModel):
    session_id: str
    question: str

class AnswerRequest(BaseModel):
    session_id: str
    answer: str

class EvaluationResponse(BaseModel):
    correct: bool
    comment: str
    question: str | None = None
    completed: bool

class HistoryResponse(BaseModel):
    session_id: str
    history: list[str]

# Helper function to get persistence for a session
def get_persistence(session_id: str) -> FileStatePersistence:
    persistence = FileStatePersistence(Path(f'sessions/question_graph_{session_id}.json'))
    persistence.set_graph_types(question_graph)
    return persistence

@app.post("/start", response_model=QuestionResponse)
async def start_session():
    """Start a new question-answer session."""
    session_id = str(uuid.uuid4())
    persistence = get_persistence(session_id)
    
    state = QuestionState()
    node = Ask()
    
    async with question_graph.iter(node, state=state, persistence=persistence) as run:
        node = await run.next()
        if isinstance(node, Answer):
            return QuestionResponse(
                session_id=session_id,
                question=node.question
            )
    
    raise HTTPException(status_code=500, detail="Failed to generate question")

@app.post("/answer", response_model=EvaluationResponse)
async def submit_answer(request: AnswerRequest):
    """Submit an answer to the current question."""
    persistence = get_persistence(request.session_id)
    
    # Load the current state
    snapshot = await persistence.load_next()
    if not snapshot:
        raise HTTPException(status_code=404, detail="Session not found or already completed")
    
    state = snapshot.state
    node = Evaluate(request.answer)
    
    async with question_graph.iter(node, state=state, persistence=persistence) as run:
        while True:
            node = await run.next()
            
            if isinstance(node, End):
                return EvaluationResponse(
                    correct=True,
                    comment=node.data,
                    completed=True
                )
            elif isinstance(node, Reprimand):
                # Continue to get next question
                continue
            elif isinstance(node, Answer):
                return EvaluationResponse(
                    correct=False,
                    comment="Incorrect. Try this question:",
                    question=node.question,
                    completed=False
                )
    
    raise HTTPException(status_code=500, detail="Failed to evaluate answer")

@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Get the history of nodes for a session."""
    persistence = get_persistence(session_id)
    
    try:
        history = await persistence.load_all()
        node_names = [e.node for e in history]
        
        return HistoryResponse(
            session_id=session_id,
            history=node_names
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    
    # Create sessions directory if it doesn't exist
    Path("sessions").mkdir(exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)