.items // []
| .[]
| "- [\(.number)] \(.title)\n  URL: \(.html_url)\n  Created: \(.created_at)  State: \(.state)\n  Labels: \([.labels[].name] | join(", "))\n  Body: \((.body // "") | gsub("[\\r\\n]+"; " ") | .[0:400])\n"
