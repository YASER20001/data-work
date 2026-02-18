# From your project root
cd "C:\Users\aalsulaimani\OneDrive - Creo Solutions\Documents\University\Capstone\Code\rifd"

# Activate venv
..\ .venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Run the full graph
python rifd_main.py

# (Optional) Test router alone
python -m router.rifd_router
