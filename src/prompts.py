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
REFINE_PROMPT_DEPRECATED = """Debug and Refine the Code: You are given a <Prompt> that describes a problem to be implemented in Python, <Faulty Trace> that the buggy implementation could not resolve the problem and fails the <Failed Test>, and the corresponding failed execution is traced and attached to code lines as comments.
You should debug according to the <Faulty Trace> to identify the root cause of its failure. 
Finally, fix the bug and wrap the refined code in between [Refined] and [/Refined].
{instruction}
{response}"""

REFINE_PROMPT = """Debug and Refine the Code: You are given a a problem to be implemented in Python, and a buggy code that tries to solve the problem but fails the test case.
You should firstly simulate the execution with the buggy code and the failed test to identify the root cause of the failure. 
Then, you should fix the bug and wrap the refined code in between [Refined] and [/Refined].
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

# v1.5 new prompts
IPT_PROMPT = """You will be given a function f and an output in the form f(??) == output. Find any input such that executing f on the input leads to the given output. There may be multiple answers, but you should only output one. In [ANSWER] and [/ANSWER] tags, complete the assertion with one such input that will produce the output when executing the function.

{instruction}
{response}"""

OPT_PROMPT = """You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

{instruction}
{response}"""