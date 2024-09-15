# NL2Code
NL2CODE_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable <Code> according to <NL_Description>

<NL_Description>
{instruction}

<Code>
{response}"""

# Used to format the `{instruction}` part above
SRC_INSTRUCT_INSTRUCTION_PROMPT = """Write a solution to the following coding problem:
{problem}"""

# Used to format src-instruct data points
SRC_INSTRUCT_ILLUSTRATION_PROMPT = """[Problem]
{problem}

[Solution]
{solution}"""


# Execution Deduction
EXEC_I_PROMPT = """Deduce the Semantic Constraints: You are given a Python program and its expected output. Find one input such that executing the program with the input leads to the given output. Complete the assertion with one such input in between [ANSWER] and [/ANSWER].
{instruction}
{response}"""

# Execution Simulation
EXEC_O_PROMPT = """Simulate the Execution: You are given a Python function and an assertion containing a function input. Complete the assertion containing the execution output corresponding to the given input in [ANSWER] and [/ANSWER] tags.
{instruction}
{response}"""


# Refinement
REFINE_PROMPT = """Debug and Refine the Code: You are given a <Prompt> that describes a problem to be implemented in Python, <Faulty Trace> that the buggy implementation could not resolve the problem and fails the <Failed Test>, and the corresponding failed execution is traced and attached to code lines as comments.
You should debug according to the <Faulty Trace> to identify the root cause of its failure. 
Finally, fix the bug and wrap the refined code in between [Refined] and [/Refined].
{instruction}
{response}"""

# Refinement with two-shot examples

REFINE_PROMPT_ONE_SHOT = """
<NL_Description>
Debug and Refine the Code: You are given a <Prompt> that describes a problem to be implemented in Python, <Faulty Trace> that the buggy implementation could not resolve the problem and fails the <Failed Test>, and the corresponding failed execution is traced and attached to code lines as comments.
You should debug according to the <Faulty Trace> to identify the root cause of its failure.
Finally, fix the bug and wrap the refined code in between [Refined] and [/Refined].

{instruction}

<Code>
{response}"""
