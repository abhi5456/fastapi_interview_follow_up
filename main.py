import os
import logging
import uvicorn
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="Interview Follow-up Question Generator",
    description="API for generating intelligent follow-up questions during interviews",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = None
def get_openai_client():
    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        openai_client = OpenAI(api_key=api_key)
    return openai_client

class FollowupRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000, description="The original interview question")
    answer: str = Field(..., min_length=10, max_length=5000, description="The candidate's response")
    role: Optional[str] = Field(None, max_length=200, description="Target role/title for context")
    interview_type: Optional[List[str]] = Field(None, description="Interview type for context")

    @validator('question')
    def question_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty or whitespace only')
        return v.strip()

    @validator('answer')
    def answer_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Answer cannot be empty or whitespace only')
        return v.strip()

    @validator('role')
    def validate_role(cls, v):
        if v is not None:
            return v.strip() if v.strip() else None
        return v

    @validator('interview_type')
    def validate_interview_type(cls, v):
        if v is not None:
            return list(set([item.strip() for item in v if item.strip()]))
        return v

class FollowupQuestionItem(BaseModel):
    followup_question: str
    # uncomment the next line to include rationale in the response
    # rationale: str

class FollowupData(BaseModel):
    questions: List[FollowupQuestionItem]

class FollowupResponse(BaseModel):
    result: str
    message: str
    data: FollowupData

class ErrorResponse(BaseModel):
    result: str
    message: str
    error_code: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

def create_system_prompt(role: Optional[str], interview_type: Optional[List[str]]) -> str:
    base_prompt = """You are an expert interview coach helping interviewers generate thoughtful follow-up questions.

Your task is to analyze the original interview question and the candidate's response, then generate 1-3 high-quality follow-up questions that:
1. Probe deeper into the candidate's experience and thought process
2. Reveal additional insights about their skills, decision-making, or approach
3. Are specific and actionable rather than generic
4. Maintain a professional, conversational tone
5. Build naturally on their response

Guidelines:
- Focus on uncovering specific examples, metrics, lessons learned, or alternative approaches
- Avoid yes/no questions
- Don't repeat information already covered
- Keep questions concise and clear
- Ensure questions are appropriate for the interview context
- Generate 1 question for simple responses, 2-3 for rich, complex responses

Safety rails:
- Never generate inappropriate, discriminatory, or illegal questions
- Avoid questions about protected characteristics (age, race, gender, religion, etc.)
- Focus only on job-relevant skills and experiences
- Maintain professional boundaries

Response format: Return ONLY a JSON array with this structure:
[
  {
    "followup_question": "Your first question here",
    "rationale": "Brief explanation of why this question is valuable (1-2 sentences)"
  },
  {
    "followup_question": "Your second question here (if applicable)",
    "rationale": "Brief explanation of why this question is valuable (1-2 sentences)"
  }
]"""

    if role:
        base_prompt += f"\n\nContext: This interview is for a {role} position."

    if interview_type:
        interview_types = ", ".join(interview_type)
        base_prompt += f"\n\nInterview type: {interview_types}"

    return base_prompt

async def generate_followup_with_openai(request: FollowupRequest) -> List[FollowupQuestionItem]:
    try:
        system_prompt = create_system_prompt(request.role, request.interview_type)
        user_message = f"""Original Question: "{request.question}"

Candidate's Answer: "{request.answer}"

Generate 1-3 thoughtful follow-up questions with rationales based on the richness and depth of the candidate's response."""
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.7,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        content = response.choices[0].message.content.strip()
        import json
        try:
            parsed_content = json.loads(content)
            questions = []
            if isinstance(parsed_content, list):
                for item in parsed_content:
                    if isinstance(item, dict) and "followup_question" in item:
                        questions.append(FollowupQuestionItem(
                            followup_question=item["followup_question"]
                        ))
            elif isinstance(parsed_content, dict) and "followup_question" in parsed_content:
                questions.append(FollowupQuestionItem(
                    followup_question=parsed_content["followup_question"]
                ))
            if questions:
                return questions
            else:
                return [FollowupQuestionItem(
                    followup_question=content
                )]
        except json.JSONDecodeError:
            return [FollowupQuestionItem(
                followup_question=content
            )]
    except openai.AuthenticationError:
        logger.error("OpenAI authentication failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OpenAI API authentication failed. Please check API key."
        )
    except openai.RateLimitError:
        logger.error("OpenAI rate limit exceeded")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="OpenAI API rate limit exceeded. Please try again later."
        )
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="OpenAI API error occurred. Please try again."
        )
    except Exception as e:
        logger.error(f"Unexpected error in OpenAI call: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating the follow-up question."
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/interview/generate-followups", response_model=FollowupResponse)
async def generate_followup_questions(request: FollowupRequest):
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OpenAI API key not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OpenAI API key not configured"
        )
    try:
        logger.info(f"Generating follow-up question for role: {request.role}, type: {request.interview_type}")
        followup_questions = await generate_followup_with_openai(request)
        response_data = FollowupData(
            questions=followup_questions
        )
        question_count = len(followup_questions)
        if question_count == 1:
            message = "Follow-up question generated."
        else:
            message = f"{question_count} follow-up questions generated."
        response = FollowupResponse(
            result="success",
            message=message,
            data=response_data
        )
        logger.info("Successfully generated follow-up question")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(result="error", message=exc.detail, error_code=str(exc.status_code)).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(_request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(result="error", message="An internal server error occurred", error_code="500").dict()
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)