"""
Clinical Report Assistant with Human-in-the-Loop Verification
Uses PydanticAI's graph workflow for structured clinical report generation
"""

from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai import Agent
from pydantic_ai.providers.google import GoogleProvider
from pydantic_graph import Graph, GraphRunContext, BaseNode, End
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import ClassVar
import sys
from pathlib import Path
from pydantic_graph.persistence.file import FileStatePersistence

# Load environment variables from .env file
load_dotenv("./.env", override=True)

class LLMSettings(BaseSettings):
    GEMINI_API_KEY: str
    GEMINI_MODEL_NAME: str
    TEMPERATURE: float

# Clinical Data Models
class PatientInfo(BaseModel):
    """Patient demographic and clinical information"""
    name: str
    age: int
    gender: str = Literal["Male", "Female", "Other"]
    chief_complaint: str
    vital_signs: Dict[str, str] = Field(description="Key vital signs with their values")
    medical_history: List[str] = Field(default_factory=list, description="Patient's medical history")
    allergies: List[str] = Field(default_factory=list, description="Known allergies")

class ClinicalAssessment(BaseModel):
    """AI generated clinical assessment"""
    diagnosis: str = Field(description="Primary diagnosis")
    alternative_diagnoses: List[str] = Field(description="List of alternative diagnoses")
    recommended_tests: List[str] = Field(description="Recommended diagnostic tests")
    treatment_plan: str = Field(description="Proposed treatment plan")
    urgency_level: Literal["Low", "Medium", "High"] = Field(description="Urgency level of the case")

class HumanVerification(BaseModel):
    """Human verification of AI generated assessment"""
    verified: bool = Field(description="Whether the assessment is verified by a clinician")
    reviewer_notes: Optional[str] = Field(description="Notes from the human reviewer")

class FinalReport(BaseModel):
    """Final clinical report after human verification"""
    patient_info: PatientInfo
    clinical_assessment: ClinicalAssessment

# Workflow State
@dataclass
class ClinicalWorkflowState:
    patient_info: Optional[PatientInfo] = None
    clinical_assessment: Optional[ClinicalAssessment] = None
    human_verification: Optional[HumanVerification] = None
    final_report: Optional[FinalReport] = None
    revision_count: int = 0
    max_revisions: int = 3

# LLM Agent
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

# Graph Nodes

@dataclass
class GenerateAssessmentNode(BaseNode[ClinicalWorkflowState]):
    """Node to generate clinical assessment using LLM"""
    _agent: ClassVar[ClinicalLLMAgent] = None

    def __post_init__(self):

        if GenerateAssessmentNode._agent is None:
            settings = LLMSettings()
            agent = ClinicalLLMAgent(settings=settings, result_type=ClinicalAssessment)

            @agent.system_prompt
            def system_prompt():
                return (
                    "You are a clinical decision support AI. Based on the provided patient information, "
                    "generate a detailed clinical assessment including diagnosis, alternative diagnoses, "
                    "recommended tests, treatment plan, and urgency level."
                )

            GenerateAssessmentNode._agent = agent

    async def run(self, ctx: GraphRunContext[ClinicalWorkflowState]) -> "HumanVerificationNode":
        state = ctx.state
        if not state.patient_info:
            raise ValueError("Patient information is required to generate clinical assessment.")
        
        prompt = (
            f"Patient Information:\n"
            f"Name: {state.patient_info.name}\n"
            f"Age: {state.patient_info.age}\n"
            f"Gender: {state.patient_info.gender}\n"
            f"Chief Complaint: {state.patient_info.chief_complaint}\n"
            f"Vital Signs: {state.patient_info.vital_signs}\n"
        )
        if ctx.state.human_verification is None:
            ctx.state.human_verification = HumanVerification(verified=False, reviewer_notes="Initial generation.")

        if not ctx.state.human_verification.verified and ctx.state.human_verification.reviewer_notes:
            prompt += f"Human Reviewer Notes: {ctx.state.human_verification.reviewer_notes}\n"

        response = await GenerateAssessmentNode._agent.run(user_prompt=prompt)
        ctx.state.clinical_assessment = response.output
        return HumanVerificationNode(assessment=response.output)
    
@dataclass
class HumanVerificationNode(BaseNode[ClinicalWorkflowState]):
    """Node for human verification of the clinical assessment"""
    assessment: ClinicalAssessment
    async def run(self, ctx: GraphRunContext[ClinicalWorkflowState]) -> "Evaluate":
       print("------Human Verification Step------")
       human_feedback = input(f"Please verify the following assessment:\n{self.assessment}\nIs this correct? (yes/no): ")
       if human_feedback.lower() == 'yes':
           verification = HumanVerification(verified=True, reviewer_notes="Verified by clinician.")
       else:
           verification = HumanVerification(verified=False, reviewer_notes="Needs revision.")
       ctx.state.human_verification = verification
       
       return Evaluate(human_feedback)

@dataclass
class Evaluate(BaseNode[ClinicalWorkflowState]):
    human_feedback: str
    _agent: ClassVar[ClinicalLLMAgent] = None
    def __post_init__(self):

        if Evaluate._agent is None:
            settings = LLMSettings()
            agent = ClinicalLLMAgent(settings=settings, result_type=HumanVerification)

            @agent.system_prompt
            def system_prompt():
                return (
                    "You are a clinical decision support AI. Based on the human feedback, "
                    "determine if the clinical assessment needs revision or is verified."
                )

            Evaluate._agent = agent

    async def run(self, ctx: GraphRunContext[ClinicalWorkflowState]) -> End[None] | "ReprimandNode":
        response = await Evaluate._agent.run(user_prompt=self.human_feedback)

        ctx.state.human_verification = response.output

        if response.output.verified:
            print("Assessment verified by human.")
            ctx.state.final_report = FinalReport(
                patient_info=ctx.state.patient_info,
                clinical_assessment=ctx.state.clinical_assessment
            )

            return End(None)
        
        else:
            return ReprimandNode()

@dataclass
class ReprimandNode(BaseNode[ClinicalWorkflowState]):
    """Node to handle reprimand and possible revision"""
    async def run(self, ctx: GraphRunContext[ClinicalWorkflowState]) -> End[None] | GenerateAssessmentNode:
        state = ctx.state
        if state.revision_count >= state.max_revisions:
            print("Maximum revisions reached. Escalating to senior clinician.")
            return End(None)
        else:
            state.revision_count += 1
            print(f"Revision {state.revision_count} of {state.max_revisions}. Regenerating assessment.")
            return GenerateAssessmentNode()
        

async def main():
    state = ClinicalWorkflowState(
        patient_info=PatientInfo(
            name="John Doe",
            age=45,
            gender="Male",
            medical_history=["Hypertension", "Diabetes"],
            allergies=["Penicillin"],
            chief_complaint="Chest pain",
            vital_signs={"BP": "120/80", "HR": "72"}
        ),
        clinical_assessment=None,
        human_verification=None,
        revision_count=0,
        max_revisions=3
    )

    human_feedback: str | None = sys.argv[1] if len(sys.argv) > 1 else None

    clinical_graph = Graph(
        nodes=(GenerateAssessmentNode, HumanVerificationNode, Evaluate, ReprimandNode),
    )

    # Use versioned persistence files to avoid conflicts
    persistence_file = Path('clinical_workflow_state_v2.json')  # Change version when schema changes
    persistence = FileStatePersistence(persistence_file)
    persistence.set_graph_types(clinical_graph)

    try:
        if snapshot := await persistence.load_next():
            state = snapshot.state
            print("Loaded existing state from persistence.")
            assert human_feedback is not None
            node = Evaluate(human_feedback)
        else:
            node = GenerateAssessmentNode()
            print("Starting new workflow.")
    except Exception as e:
        print(f"Error loading persistence file: {e}")
        print("Starting fresh workflow - you may want to backup/migrate your old data")
        node = GenerateAssessmentNode()
        # Optionally clear the corrupted file
        # Path('clinical_workflow_state.json').unlink(missing_ok=True)

    async with clinical_graph.iter(node, state=state, persistence=persistence) as run:
        while True:
            node_or_end = await run.next()
            print('Node:', node_or_end)

            if isinstance(node_or_end, End):
                print("Final Report or Escalation:", node_or_end.data)
                break
            elif isinstance(node_or_end, HumanVerificationNode):
                break

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())