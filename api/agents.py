import re
from typing import Optional, List, Dict, Any

from langchain_openai import ChatOpenAI

# New imports for the agents
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_community.callbacks import get_openai_callback
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os

import schema
import rag_agent
import consultation_agent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

db: Optional[SQLDatabase] = None

# Create and return SQL agent
def get_sql_agent(llm):
    """Creates a LangChain SQL Agent."""
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent_executor

def get_chat_agent(llm: ChatOpenAI, history: List[dict]):
    """
    Creates a new agent for general chat, with guardrails to keep the conversation on topic.
    """
    guardrail_system_prompt = """You are "JobIndo AI", a helpful and professional career assistant for the Indonesian market.

### YOUR ROLE:
- Engage in helpful conversation about job searching, career growth, and the Indonesian job market.
- Use the conversation history to provide context-aware responses.
- **Handoff Awareness:** If a user asks a complex question that clearly requires specific data (e.g., "Find me jobs in Bali", "What is the average salary?") or analysis (e.g., "Review my CV"), acknowledge the request and inform them that you are ready to help, but do not try to invent specific database data yourself. The system will handle specific routing in the next turn if they are specific.
- **Guardrails:** Politely decline off-topic discussions (e.g., politics, entertainment, cooking) by pivoting back to career topics.

### TONE:
- Professional, encouraging, and concise.
- **Language:** Respond in the same language as the user (Indonesian or English).
"""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=guardrail_system_prompt),
        *history,
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain


# Chat function that acts as the MAIN AGENT
def chat(req: schema.ChatRequest, cv_file_contents: Optional[str]) -> Dict[str, Any]:
    """
    This is the main agent that routes questions to the appropriate specialist agent.
    It now returns a dictionary with the response and usage data.
    """
    user_question = req.message.content
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
    )

    # Default response in case of errors
    error_response = {
        "content": "Sorry, I encountered an error. Please try again.",
        "agent_used": "Error",
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    # Convert message history to LangChain format for conversational context
    lc_history = []
    if req.history:
        for msg in req.history:
            if msg.role == "human":
                lc_history.append(HumanMessage(content=msg.content))
            elif msg.role == "ai":
                lc_history.append(AIMessage(content=msg.content))

    # --- Agent States & Multi-turn Conversation ---
    # Check if the last message was the AI asking for a job description
    is_waiting_for_jd = False
    if req.history:
        last_message = req.history[-1]
        if last_message.role == 'ai' and "please paste the job description" in last_message.content.lower():
            is_waiting_for_jd = True

    # If waiting for a job description, go straight to the ResumeAgent.
    if is_waiting_for_jd:
        if not cv_file_contents:
            return {
                "content": "Error: CV content is missing. Please upload your CV again before pasting the job description.",
                "agent_used": "ResumeAgent",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        print(f"[{req.session_id}] --- Executing Resume Analysis ---")
        resume_agent = ResumeAgent(llm)
        job_description = user_question

        with get_openai_callback() as cb:
            analysis_result = resume_agent.run(cv_text=cv_file_contents, job_description=job_description)

            if "error" in analysis_result:
                return {
                    "content": analysis_result["error"],
                    "agent_used": "ResumeAgent",
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                }

            # Format the response
            response_content = f"""
### Resume Analysis Complete

Here is your resume optimization report:

**ATS Score Analysis:**
- **Before:** {analysis_result['before_score']}/100
- **After:** {analysis_result['after_score']}/100

---

### Rewritten Resume (Optimized for ATS & Recruiters)

{analysis_result['rewritten_cv']}
"""
            return {
                "content": response_content.strip(),
                "agent_used": "ResumeAgent",
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
            }

    # 1. Router: Decide which agent to use
    router_prompt_template = """You are the intelligent central router for the "JobIndo" AI Assistant. Your sole purpose is to analyze the user's latest message and conversation history to route the request to the most appropriate specialized agent.

### AVAILABLE AGENTS & DEFINITIONS:

1. **SQL Agent (SQL)**
   - **Trigger:** Queries asking for *structured data*, *statistics*, or *lists* based on specific criteria from the database.
   - **Keywords:** "how many", "average salary", "list jobs in [Location]", "top paying", "filter by", "count".
   - **Examples:** "Show me all Data Scientist jobs in Jakarta", "What is the average salary for Python developers?", "Count the number of remote jobs."

2. **Resume Agent (RESUME)**
   - **Trigger:** Requests involving the user's Curriculum Vitae (CV) or Resume. Includes uploading, reviewing, scoring, or rewriting.
   - **Keywords:** "CV", "resume", "rewrite", "optimize", "ATS", "score my resume", "upload".
   - **Examples:** "Here is my CV, please review it.", "Rewrite my resume for this job description.", "What is my ATS score?"

3. **Consultation Agent (CONSULTATION)**
   - **Trigger:** Broad, high-level career advice, roadmap planning, skill gap analysis, or requests for a "full consultation".
   - **Keywords:** "career path", "what should I learn", "career advice", "roadmap", "skills for [Role]", "consultation", "recommend a job".
   - **Examples:** "I want to become a Data Engineer, what is the roadmap?", "Give me a career consultation based on my profile.", "What skills do I lack?", "Recommend jobs for me."

4. **RAG Agent (RAG)**
   - **Trigger:** Deep semantic questions about the *content* of specific job descriptions or qualitative analysis.
   - **Keywords:** "details about this job", "what does this role entail", "explain the requirements", "summarize this job".
   - **Note:** If the user asks to *find* jobs, prefer SQL (if specific criteria) or Consultation (if broad recommendation). Use RAG mainly for analyzing *content* or specific details of a retrieved job.

5. **Chat Agent (CHAT)**
   - **Trigger:** General conversation, greetings, clarifications, or questions about the chat history itself.
   - **Keywords:** "hello", "who are you", "thank you", "explain that again".
   - **Default:** If the query does not clearly fit the others, route to CHAT.

### ROUTING INSTRUCTIONS:
- Analyze the `<conversation_history>` to understand context (e.g., if the user just uploaded a CV, they likely want RESUME or CONSULTATION).
- Analyze the `<user_question>`.
- **Reasoning:** Think silently about which agent fits best.
- **Output:** Return ONLY the agent name: "SQL", "RESUME", "CONSULTATION", "RAG", or "CHAT".

<conversation_history>
{history}
</conversation_history>

<user_question>
{question}
</user_question>
"""
    router_prompt = PromptTemplate(template=router_prompt_template, input_variables=["history", "question"])
    router_chain = router_prompt | llm | StrOutputParser()

    history_str = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in req.history or []])

    try:
        with get_openai_callback() as cb:
            route = router_chain.invoke({"question": user_question, "history": history_str})
            # This callback only counts the tokens for the router itself. We will wrap each agent call below.
            print(f"[{req.session_id}] Router decided: {route} (tokens: {cb.total_tokens})")

        if "sql" in route.lower():
            print(f"[{req.session_id}] --- Activating SQL Agent ---")
            sql_agent = get_sql_agent(llm)
            agent_input = {"input": user_question, "chat_history": lc_history}
            with get_openai_callback() as sql_cb:
                agent_response = sql_agent.invoke(agent_input)
                return {
                    "content": agent_response.get("output", "Sorry, I could not find an answer."),
                    "agent_used": "SQLAgent",
                    "prompt_tokens": sql_cb.prompt_tokens,
                    "completion_tokens": sql_cb.completion_tokens,
                }
        elif "resume" in route.lower():
            print(f"[{req.session_id}] --- Activating Resume Agent (Initial Request) ---")
            if not cv_file_contents:
                content = "It looks like you want to work on your resume. Please upload your CV first using the button on the sidebar."
            else:
                content = "Resume agent is active. Please paste the job description you are targeting, and I will begin the analysis."
            return {
                "content": content,
                "agent_used": "ResumeAgent",
                "prompt_tokens": 0, # No LLM call yet
                "completion_tokens": 0,
            }
        elif "rag" in route.lower():
            print(f"[{req.session_id}] --- Activating RAG Agent ---")
            # Assuming rag_agent.ask_job_question is modified to return a dict with tokens
            with get_openai_callback() as rag_cb:
                response_content = rag_agent.ask_job_question(user_question, history_str, req.session_id)
                return {
                    "content": response_content,
                    "agent_used": "RAGAgent",
                    "prompt_tokens": rag_cb.prompt_tokens,
                    "completion_tokens": rag_cb.completion_tokens,
                }
        elif "consultation" in route.lower():
            print(f"[{req.session_id}] --- Activating Consultation Agent ---")
            with get_openai_callback() as consultation_cb:
                if cv_file_contents:
                    question = ("<cv_contents>\n" +
                                cv_file_contents +
                                "</cv_contents>\n\n<user_question>" +
                                user_question +
                                "</user_question>")
                else:
                    question = user_question
                response_content = consultation_agent.career_consultation_agent(question)
                return {
                    "content": response_content,
                    "agent_used": "ConsultationAgent",
                    "prompt_tokens": consultation_cb.prompt_tokens,
                    "completion_tokens": consultation_cb.completion_tokens,
                }
        else: # Default to CHAT agent
            print(f"[{req.session_id}] --- Activating Chat Agent ---")
            chat_agent = get_chat_agent(llm, lc_history)
            with get_openai_callback() as chat_cb:
                response_content = chat_agent.invoke({"question": user_question})
                return {
                    "content": response_content,
                    "agent_used": "ChatAgent",
                    "prompt_tokens": chat_cb.prompt_tokens,
                    "completion_tokens": chat_cb.completion_tokens,
                }
    except Exception as e:
        print(f"[{req.session_id}] An error occurred in the main agent: {e}")
        return error_response

class ResumeAgent:
    """
    An agent dedicated to optimizing resumes for ATS and human recruiters.
    """
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.action_verbs = [
            'achieved', 'accelerated', 'administered', 'advised', 'advocated', 'analyzed', 'authored', 'automated',
            'balanced', 'boosted', 'budgeted', 'built', 'calculated', 'centralized', 'chaired', 'clarified',
            'collaborated', 'conceived', 'conceptualized', 'coordinated', 'created', 'customized', 'decreased',
            'defined', 'designed', 'developed', 'directed', 'doubled', 'drove', 'eliminated', 'enabled', 'enforced',
            'engineered', 'enhanced', 'established', 'evaluated', 'executed', 'expanded', 'facilitated', 'formalized',
            'formed', 'formulated', 'founded', 'generated', 'grew', 'halved', 'headed', 'identified', 'implemented',
            'improved', 'increased', 'initiated', 'inspired', 'instituted', 'integrated', 'introduced', 'invented',
            'launched', 'led', 'lowered', 'managed', 'mastered', 'maximized', 'measured', 'mentored', 'minimized',
            'modernized', 'motivated', 'negotiated', 'optimized', 'orchestrated', 'overhauled', 'oversaw', 'pioneered',
            'planned', 'predicted', 'prepared', 'presented', 'prioritized', 'produced', 'proposed', 'proved',
            'provided', 'raised', 'rebuilt', 'recommended', 'redesigned', 'reduced', 're-engineered', 'refined',
            'reformed', 'reorganized', 'replaced', 'restructured', 'revamped', 'saved', 'scaled', 'scheduled',
            'secured', 'selected', 'simplified', 'slashed', 'solved', 'spearheaded', 'specified', 'standardized',
            'streamlined', 'strengthened', 'structured', 'succeeded', 'supervised', 'surpassed', 'systematized',
            'targeted', 'taught', 'tested', 'trained', 'transformed', 'tripled', 'unified', 'updated', 'upgraded',
            'validated', 'verbalized', 'verified', 'visualized', 'won'
        ]

    def _extract_keywords_from_jd(self, job_description: str) -> List[str]:
        """Extracts keywords from a job description using an LLM."""
        print("ResumeAgent: Extracting keywords from JD...")
        prompt_template = """
        You are an expert HR analyst. Your task is to extract the most important keywords from a job description.
        Focus on specific, marketable skills, technologies, and essential qualifications. Ignore generic phrases.
        Please return a simple comma-separated list of the top 10-15 keywords.

        Example:
        Job Description: "We are looking for a Senior Software Engineer with experience in Python, Django, and React. The ideal candidate will have a strong understanding of cloud services like AWS and a background in CI/CD pipelines."
        Output: "Python, Django, React, AWS, CI/CD, Senior Software Engineer"

        Job Description:
        "{job_description}"

        Output:
        """
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({"job_description": job_description})
            keywords = [keyword.strip() for keyword in result.split(',') if keyword.strip()]
            print(f"ResumeAgent: Extracted keywords: {keywords}")
            return keywords
        except Exception as e:
            print(f"ResumeAgent: Error extracting keywords: {e}")
            return []

    def _calculate_ats_score(self, cv_text: str, jd_keywords: List[str]) -> int:
        """Calculates a heuristic-based ATS score out of 100."""
        print("ResumeAgent: Calculating ATS score...")
        score = 0

        # 1. Keyword Matching (70 points)
        if jd_keywords:
            matches = 0
            cv_lower = cv_text.lower()
            for keyword in jd_keywords:
                if keyword.lower() in cv_lower:
                    matches += 1
            keyword_score = (matches / len(jd_keywords)) * 70
            score += keyword_score
            print(f"ResumeAgent: Keyword score: {keyword_score:.2f}/70")

        # 2. Quantitative Results (20 points)
        # Finds numbers, percentages, dollar amounts
        quant_results = re.findall(r'(\d+%|\$\d+|\d{2,})', cv_text)
        quant_score = min(len(quant_results) * 2, 20)
        score += quant_score
        print(f"ResumeAgent: Quantitative score: {quant_score}/20")

        # 3. Action Verbs (10 points)
        action_verb_matches = 0
        cv_words = set(re.findall(r'\b\w+\b', cv_text.lower()))
        found_verbs = {verb for verb in self.action_verbs if verb in cv_words}
        action_verb_score = min(len(found_verbs), 10)
        score += action_verb_score
        print(f"ResumeAgent: Action verb score: {action_verb_score}/10")

        final_score = min(int(score), 100)
        print(f"ResumeAgent: Final Score: {final_score}/100")
        return final_score

    def _rewrite_resume_with_llm(self, cv_text: str, jd_keywords: List[str]) -> str:
        """Rewrites the resume to be more impactful and keyword-rich."""
        print("ResumeAgent: Rewriting resume with LLM...")

        prompt_template = """
        You are a world-class career coach and professional resume writer, an expert at optimizing resumes for both Applicant Tracking Systems (ATS) and human recruiters.

        Your task is to rewrite the provided resume to be significantly more impactful and tailored for a specific job description.

        Here is the original resume:
        ---
        {cv_text}
        ---

        Here is a list of the most important keywords extracted from the target job description:
        ---
        {keywords}
        ---

        ### INSTRUCTIONS:
        1.  **Language Detection:** Detect the language of the original resume (Indonesian or English). **You MUST output the rewritten resume in the SAME language as the original**, unless the user explicitly asked to translate it.
        2.  **Integrate Keywords:** Naturally and seamlessly weave the provided keywords into the resume. Do not just list them. They should fit the context of the experience and skills sections.
        3.  **Quantify Impact:** This is your most important task. Transform experience bullet points from simple duties into powerful, measurable achievements. Use metrics, numbers, percentages, and dollar amounts to show impact.
        4.  **Use Placeholders for Missing Numbers:** If the original resume lacks numbers, invent realistic and specific placeholders and enclose them in square brackets. For example, rewrite "Managed a team" to "Managed a team of [5] engineers" or "Improved performance" to "Improved system performance by [~20%]". This is crucial for creating a high-impact resume.
        5.  **Action Verbs:** Start every bullet point with a strong action verb.
        6.  **Preserve Structure:** Maintain the original section layout of the resume (e.g., Summary, Experience, Education, Skills).
        7.  **Return Only the Resume:** Your final output should ONLY be the full text of the rewritten, optimized resume. Do not include any commentary, greetings, or explanations before or after the resume text.

        Now, begin the rewrite.
        """

        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()

        try:
            # Convert list of keywords to a comma-separated string for the prompt
            keywords_str = ", ".join(jd_keywords)
            rewritten_cv = chain.invoke({"cv_text": cv_text, "keywords": keywords_str})
            print("ResumeAgent: Successfully rewrote resume.")
            return rewritten_cv
        except Exception as e:
            print(f"ResumeAgent: Error rewriting resume: {e}")
            return "Error: Could not rewrite the resume at this time."

    def run(self, cv_text: str, job_description: str) -> Dict[str, any]:
        """
        Runs the full resume analysis and rewrite workflow.
        """
        print("ResumeAgent: Starting analysis...")

        # 1. Extract keywords from the job description
        jd_keywords = self._extract_keywords_from_jd(job_description)

        if not jd_keywords:
            return {
                "error": "Could not extract keywords from the job description. Please try again with a different one."
            }

        # 2. Calculate the "before" score
        before_score = self._calculate_ats_score(cv_text, jd_keywords)

        # 3. Rewrite the resume
        rewritten_cv = self._rewrite_resume_with_llm(cv_text, jd_keywords)

        # 4. Calculate the "after" score
        after_score = self._calculate_ats_score(rewritten_cv, jd_keywords) # No longer a placeholder increase

        print("ResumeAgent: Analysis complete.")

        return {
            "before_score": before_score,
            "after_score": after_score,
            "rewritten_cv": rewritten_cv,
        }