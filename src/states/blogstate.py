from typing import TypedDict
from pydantic import BaseModel,Field

class Blog(BaseModel):
    title:str=Field(description="the title of the blog post")
    content:str=Field(description="The main content of the blog post")

class BlogState(TypedDict, total=False):
    topic: str
    current_language: str
    blog: dict
    translated_title: str
    translated_content: str

