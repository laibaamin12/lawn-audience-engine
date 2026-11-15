# Lawn Audience Engine

Simulated-data demo of an audience analytics engine for a women's lawn campaign (11.11, 12.12, Eid, winter launch).

Run in order (Python 3.10+, pandas, numpy, scikit-learn, scipy):

    python 01_generate_data.py   # posts.csv, comments.csv   (simulated; data_source = simulated)
    python 02_analyse.py         # results.json, posts_scored.csv, comments_clustered.csv
    python 03_build_site.py      # index.html  (open in any browser, no server needed)

To use real data: replace posts.csv and comments.csv with collected records using the same columns,
set data_source = observed, and extend the LEX seed lexicon in 02_analyse.py with new slang.
Collect comments only through methods the platform's terms allow.
