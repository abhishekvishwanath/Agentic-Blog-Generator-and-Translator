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
        LLMs sometimes return a full blog even when asked for a title.
        This normalizes to a single-line, human-friendly title.
        """
        if not raw_text:
            return ""

        # Take first non-empty line.
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        first = lines[0] if lines else raw_text.strip()

        # Strip common markdown/title wrappers.
        first = first.lstrip("#").strip()
        if (first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'")):
            first = first[1:-1].strip()

        # If model returned "Title: ...", remove the label.
        lowered = first.lower()
        if lowered.startswith("title:"):
            first = first.split(":", 1)[1].strip()

        return first

    def _parse_translated_title_content(self, raw_text: str) -> tuple[str, str]:
        """
        Expects model output in the format:
        TITLE: ...
        CONTENT:
        ...
        Falls back gracefully if format isn't followed.
        """
        if not raw_text:
            return "", ""

        text = raw_text.strip()
        lower = text.lower()

        if "title:" in lower and "content:" in lower:
            # Find first occurrences (case-insensitive) while slicing original text
            t_idx = lower.find("title:")
            c_idx = lower.find("content:")
            if t_idx != -1 and c_idx != -1 and c_idx > t_idx:
                title_part = text[t_idx + len("title:") : c_idx].strip()
                content_part = text[c_idx + len("content:") :].strip()
                return self._extract_title(title_part), content_part

        # Fallback: use first line as title and the rest as content.
        lines = [ln.rstrip() for ln in text.splitlines()]
        first_non_empty = ""
        first_i = 0
        for i, ln in enumerate(lines):
            if ln.strip():
                first_non_empty = ln.strip()
                first_i = i
                break
        title = self._extract_title(first_non_empty)
        content = "\n".join(lines[first_i + 1 :]).lstrip() if len(lines) > first_i + 1 else ""
        return title, content

    def title_creation(self,state:BlogState):
        """
        create the title for the blog
        """

        if "topic" in state and state["topic"]:
            prompt = """
                   You are an expert blog content writer.
                   Return ONLY a single blog title for the topic: {topic}
                   Constraints:
                   - One line only
                   - No markdown headings (#), no quotes
                   - Do NOT include the blog body/content
                   """
            
            sytem_message=prompt.format(topic=state["topic"])
            response=self.llm.invoke(sytem_message)

            title = self._extract_title(getattr(response, "content", "") or str(response))
            return {"blog": {"title": title}}
        
    def content_generation(self,state:BlogState):
        if "topic" in state and state["topic"]:
            system_prompt = """You are expert blog writer. Use Markdown formatting.
            Generate a detailed blog content with detailed breakdown for the {topic}.
            IMPORTANT:
            - Do NOT include a separate title line (title is handled elsewhere)
            - Start directly with the content sections in markdown
            """

            system_message = system_prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)

            return {"blog": {"title": state['blog']['title'], "content": response.content}}

    def no_translation(self, state: BlogState):
        """
        For English (or unsupported languages), pass-through without translation.
        """
        return {"blog": state.get("blog", {})}
        
    def translation(self,state:BlogState):
        """
        Translate the content to the specified language.
        """
        translation_prompt = """
        You are a professional translator.
        Translate BOTH the blog TITLE and CONTENT into {current_language}.

        Rules:
        - Maintain the original tone, style, and formatting.
        - Adapt cultural references and idioms to be appropriate for {current_language}.
        - Return output in EXACTLY this format:
          TITLE: <translated title>
          CONTENT:
          <translated content>
        - Do not add any extra commentary/notes.

        ORIGINAL TITLE:
        {blog_title}

        ORIGINAL CONTENT:
        {blog_content}
        """

        blog_title = state.get("blog", {}).get("title", "")
        blog_content = state.get("blog", {}).get("content", "")
        messages = [
            HumanMessage(
                translation_prompt.format(
                    current_language=state["current_language"],
                    blog_title=blog_title,
                    blog_content=blog_content,
                )
            )
        ]
        response = self.llm.invoke(messages)
        translated_title, translated_content = self._parse_translated_title_content(
            getattr(response, "content", "") or str(response)
        )
        return {
            "blog": {
                "title": translated_title or blog_title,
                "content": translated_content or (getattr(response, "content", "") or str(response)),
            }
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
        elif state["current_language"] == "english":
            return "english"
        else:
            return "english"