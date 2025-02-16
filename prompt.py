INTERPRET_TASK_PROMPT = """
You will be provided some text data. This text will contain a task description. Beware! This description could be in any language. You have to interpret and return one of the following json objects:

1. {"category": "install uv library and run datagen", "argument": ${email that you found in the user prompt}}
2. {"category": "format using prettier"}
3. {"category": "count wednesdays"}
4. {"category": "sort array of contacts"}
5. {"category": "write 10 most recent logs"}
6. {"category": "find markdown files and extract h1 tags"}
7. {"category": "extract sender email address"}
8. {"category": "extract credit card number"}
9. {"category": "find similar comments"}
10. {"category": "find total sales of 'gold' ticket type in the db table"}
11. fetch data from api
12. clone a git repo and make a commit
13. run a sql query in sqlite
14. run a sql query in duckdb
15. extract data from a website
16. compress/resize image
17. transcribe audio from an mp3 file
18. convert markdown to html
19. api endpoint to filter csv file and return json
20. {"category": "not a valid task"}

Return just the json object without the index number and any other text.
For example, if the task description is "Run the datagen script with email argument as 'example@email.com'", you should return
{"category": "install uv library and run datagen", "argument": "example@email.com"}
"""