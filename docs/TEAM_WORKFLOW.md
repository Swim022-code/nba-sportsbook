# Team workflow (GitHub collaboration)

The rubric asks for **meaningful commit history from each team member**. Each person should make
their own commits from their own GitHub account. Here is a simple flow that takes about 15 minutes per person.

## One-time setup (each member)
1. The repo owner adds everyone: GitHub repo → **Settings → Collaborators → Add people**.
2. Each member:
   ```bash
   git clone https://github.com/<owner>/nba-sportsbook.git
   cd nba-sportsbook
   git config user.name "Your Name"
   git config user.email "the-email-on-your-github-account"
   python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```

## Making a change (feature branch → pull request)
```bash
git checkout -b yourname/short-description
# ...edit files...
pytest -q                 # make sure tests still pass
git add <files>
git commit -m "Clear description of what you changed"
git push -u origin yourname/short-description
```
Then on GitHub: **Compare & pull request** → a teammate reviews → **Merge**. CI runs automatically.
Afterwards: `git checkout main && git pull`.

## Suggested real contributions (one or more per person)
Pick tasks that match each person's role in the README team table:

| Member | Ideas |
|---|---|
| Data | Add a "Last 10 games" form column to the Teams tab; add attendance analysis to the EDA |
| EDA | Add a chart (e.g. rebounds vs. win % by team); write the "Key insights" in your own words |
| Model | Tune `HALF_LIFE_DAYS` / `RIDGE_LAMBDA` in `src/odds_model.py` and report the back-test change |
| LLM / UI | Add a new example question to the chatbot; improve a prompt in `src/llm.py` |
| Everyone | Fill in your row of the README team table; write your part of the report's collaboration section |

Each of these is small but real, and shows up in the GitHub history under your own name.
