"""Step 3: inject results.json into template.html -> index.html (self-contained website)."""
import json
data = open("results.json", encoding="utf-8").read().replace("</", "<\\/")
html = open("template.html", encoding="utf-8").read().replace("__DATA__", data)
open("index.html", "w", encoding="utf-8").write(html)
print(len(html) // 1024, "KB")
