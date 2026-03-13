from src.states.blogstate import BlogState
from langchain_core.messages import SystemMessage, HumanMessage
from src.states.blogstate import Blog

class BlogNode:
    """
    A class to represent he blog node
    """

    def __init__(self,llm):
        self.llm=llm
    
    def _extract_title(self, raw_text: str) -> str:
        """
        Normalize an LLM response to a single-line, human-friendly title.
        """
        if not raw_text:
            return ""

        # Take first non-empty line
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        first = lines[0] if lines else raw_text.strip()

        # Strip common markdown/title wrappers
        first = first.lstrip("#").strip()
        if (first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'")):
            first = first[1:-1].strip()

        # If model returned "Title: ...", remove the label.
        lowered = first.lower()
        if lowered.startswith("title:"):
            first = first.split(":", 1)[1].strip()

        return first

    def title_creation(self,state:BlogState):
        """
        create the title for the blog
        """

        if "topic" in state and state["topic"]:
            prompt="""
                   You are an expert blog content writer. Use Markdown formatting. Generate
                   a blog title for the {topic}. This title should be creative and SEO friendly
                   """
            
            sytem_message=prompt.format(topic=state["topic"])
            response=self.llm.invoke(sytem_message)

            clean_title = self._extract_title(getattr(response, "content", "") or str(response))
            return {"blog":{"title":clean_title}}
        
    def content_generation(self,state:BlogState):
        if "topic" in state and state["topic"]:
            system_prompt = """You are expert blog writer. Use Markdown formatting.
            Generate a detailed blog content with detailed breakdown for the {topic}"""

            system_message = system_prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)

            return {"blog": {"title": state['blog']['title'], "content": response.content}}
        
    def translation(self,state:BlogState):
        """
        Translate the content to the specified language.
        """
        translation_prompt="""
        Translate the following content into {current_language}.
        - Maintain the original tone, style, and formatting.
        - Adapt cultural references and idioms to be appropriate for {current_language}.

        ORIGINAL CONTENT:
        {blog_content}

        """
        
        original_title = state["blog"].get("title", "")
        original_content = state["blog"]["content"]
        blog_content = original_content
        messages = [
            HumanMessage(
                translation_prompt.format(
                    current_language=state["current_language"],
                    blog_content=blog_content,
                )
            )
        ]

        # Translate full content
        response = self.llm.invoke(messages)

        # Separately translate the title to target language
        title_translation_prompt = """
        Translate the following blog title into {current_language}.
        Rules:
        - Return ONLY the translated title
        - One line only
        - No markdown headings, no quotes

        TITLE: {blog_title}
        """
        title_msg = title_translation_prompt.format(
            current_language=state["current_language"],
            blog_title=original_title,
        )
        title_response = self.llm.invoke(title_msg)
        translated_title = self._extract_title(getattr(title_response, "content", "") or str(title_response))

        return {
            # Preserve original (English) blog
            "blog": {
                "title": original_title,
                "content": original_content,
            },
            # Add translated artifacts at top level
            "translated_title": translated_title or original_title,
            "translated_content": response.content,
        }

    def route(self, state: BlogState):
        return {"current_language": state['current_language'] }
    

    def route_decision(self, state: BlogState):
        """
        Route the content to the respective translation function.
        """
        if state["current_language"] == "hindi":
            return "hindi"
        elif state["current_language"] == "french": 
            return "french"
        else:
            # Default to french translation if an unsupported language is provided
            return "french"