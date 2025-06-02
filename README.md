pip install -e .
rpl init
rpl log "Experiment X"


# Initialize a project
./rpl.py init optics --description "Permittivity experiments"

# Log an experiment
./rpl.py log --title "Day 1 test" --notes "Measured permittivity" --tags "GHz,sapphire"

# Upload a file
./rpl.py upload results_day1.pdf

# Query the project knowledge base
./rpl.py query "What did we learn about sapphire?"

# See current project
./rpl.py current