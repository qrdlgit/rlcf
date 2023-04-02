# rlcf
Reinforcement Learning Computer Feedback

Basic idea is 

1. Using prompt which asks for changes to improve an ML program + MLs programs, generate responses from codebert model
2. Using generated responses, make changes in ML program and evaluate how well the program compiles/runs/has improved
3. Use the evaluation from 2 as a 'reward' model for a RLCF (Reinforcement Learning Computer Feedback) and fine tune codebert model
4. Loop to 1

This is all WIP, missing a few key pieces, but I am actively working on it so should have it up soon.  Suggestions welcomed.

To learn more, follow me on twitter:  https://twitter.com/QRDL
