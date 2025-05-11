import typer
import json
import os

db = "db.json"
app = typer.Typer()

@app.command()
def add(tn: str,tp: str):
    tasks=[]
    task = {
        "task_name":tn,
        "task_progress":tp
    }
    
    if os.path.exists(db):
        with open(db,"r") as f:
            tasks=json.load(f)

    tasks.append(task)

    with open(db, "w") as f:
        json.dump(tasks, f,indent=4)

    print(f"Task '{task}' saved to {db}.")


@app.command()
def list():
    with open(db, "r") as f:
        tasks = json.load(f)

    typer.secho("\nto-do list:\n", fg=typer.colors.BLUE, bold=True) 
    columns=(
        "ID. ",
        "| Name    ",
        "| Progress    ",
    )
    headers = "".join(columns)
    typer.secho(headers, fg=typer.colors.BLUE, bold=True)
    typer.secho("-" * len(headers), fg=typer.colors.BLUE, bold=True)

    for task in tasks:
        typer.secho(
            f"{tasks.index(task)+1}{(len(columns[0]) - len(str(tasks.index(task)))-1) * ' '}"
            f" | {task['task_name']}{(len(columns[1]) - len(str(task['task_name']))-3) * ' '}"
            f" | {task['task_progress']}",#"{(len(columns[2]) - len(str(task['task_progress']))) * ' '}",
            fg=typer.colors.GREEN,
            bold=True,
        )
    typer.secho("-" * len(headers), fg=typer.colors.BRIGHT_RED, bold=True)


if __name__ == "__main__":
    app()