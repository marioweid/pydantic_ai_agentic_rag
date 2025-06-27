from dataclasses import dataclass
import re
import unicodedata

def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """Slugify a string, to make it URL friendly."""
    # Taken unchanged from https://github.com/Python-Markdown/markdown/blob/3.7/markdown/extensions/toc.py#L38
    if not unicode:
        # Replace Extended Latin characters with ASCII, i.e. `žlutý` => `zluty`
        value = unicodedata.normalize("NFKD", value)
        value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(rf"[{separator}\s]+", separator, value)

@dataclass
class DocsSection:
    id: int
    parent: int | None
    path: str
    level: int
    title: str
    content: str

    def url(self) -> str:
        url_path = re.sub(r"\.md$", "", self.path)
        return (
            f"https://logfire.pydantic.dev/docs/{url_path}/#{slugify(self.title, '-')}"
        )

    def embedding_content(self) -> str:
        return "\n\n".join((f"path: {self.path}", f"title: {self.title}", self.content))