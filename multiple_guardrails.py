# Import Guard and Validator
from guardrails.hub import BanList
from guardrails import Guard

# Setup Guard
guard = Guard().use(
    BanList(banned_words=['codename','athena'])
)

guard.validate("Hello world! I really like Python.")  # Validator passes
guard.validate("I am working on a project with the code name A T H E N A")  # Validator fails