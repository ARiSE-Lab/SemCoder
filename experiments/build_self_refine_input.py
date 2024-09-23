import argparse
import json
from experiments.construct_trace import attach_trace_to_code
import re

def extract_failed_test(traced_program, funcname):
    """
    extract the failed test case from the traced program
    """
    failed_test = ""
    for line in traced_program.split('\n'):
        if line.lstrip().startswith(f'retv = {funcname}'):
            if line.split(",")[-1] == " )":
                line = ",".join(line.split(",")[:-1]) + ")"
            failed_test += line.lstrip() + '\n'
        
        if line.lstrip().startswith('assert retv == '):
            failed_test += line.lstrip() + '\n'
    return failed_test

def extract_trace(trace, inference=False):
    """
    extract the trace of the real function, remove the other utility information
    """
    trace_lines = trace.split('\n')
    input_pattern = re.compile(r"Starting var:\.* (.*?) = (.*)")
    # make sure only one block has indentation. # TODO: Sanity check regarding this heuristic
    lines_w_indent = []
    for idx, line in enumerate(trace_lines):
        if line.startswith('    '):
            lines_w_indent.append(idx)
            if input_match := input_pattern.match(line.lstrip()):
                input_info = input_match.group(2)
                # we do not consider inputs with ..., since the information is not complete
                if '...' in input_info and not inference:
                    return None
                if len(input_info) > 100 and not inference:
                    return None
    # check whether the line numbers are continuous
    if len(lines_w_indent) == 0:
        if not inference:
            return None
        else:
            # print(trace)
            return trace
    end = -1
    for i in range(len(lines_w_indent) - 1):
        if lines_w_indent[i] + 1 != lines_w_indent[i + 1]:
            if not inference:
                return None
            else:
                # print(trace)
                end = lines_w_indent[i]
    # extract the trace lines with indentation
    start = lines_w_indent[0]
    if end == -1:
        end = lines_w_indent[-1]
    trace = '\n'.join([l.lstrip() for l in trace_lines[start:end + 1]])
    return trace

def attach_execution_exception(traced_program, trace):
    """
    attach the exception happening during the execution (i.e., not the test-case assertion error) to the traced program
    """
    exception_pattern = re.compile(r"Exception:\.* (.*)")
    current_line = -1
    code_lines = traced_program.split('\n')
    for line in trace.split('\n'):
        line = line.strip()
        line_split = line.split()
        if len(line_split) < 3:
            continue
        if line_split[1] == "exception" and line_split[2].isdigit():
            current_line = int(line_split[2])
        if exception_match := exception_pattern.match(line):
            exception_info = exception_match.group(1)
            # attach the exception to the code
            if current_line != -1:
                code_lines[current_line - 1] += f' # [EXCEPTION] {exception_info} [/EXCEPTION]'
    return '\n'.join(code_lines)

def clean_code(inlined_trace):
    """
    clean the code, remove the pysnooper decorator and other stuff
    """
    cleaned_lines = []
    for line in inlined_trace.split('\n'):
        if line.lstrip().startswith('@pysnooper.snoop'):
            break
        if "pysnooper" in line:
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def prepare_data():
    """
    Prepare the data for the self-refinement rational generation
    """
    total_samples = 0
    no_failed_test = 0
    no_func_trace = 0
    dense_samples = 0
    sample_per_example = {}
    with open(args.output_file, 'w') as f_out:
        with open(args.bug_report_file, 'r') as f:
            for line in f:
                bug = json.loads(line)
                nl = bug['instruction']
                solution = bug['func']
                raw_index = bug['raw_index']
                buggy_responses = bug['buggy_responses']
                for br in buggy_responses:
                    funcname = br['funcname']
                    key = solution + br["func"]
                    # each br is a buggy response: the model generates top-k sequences
                    # if br["sim_to_ref"] < 75:
                    #     continue
                    failures = br['failures']
                    for fail_info in failures:
                        if key not in sample_per_example:
                            sample_per_example[key] = 1
                        if sample_per_example[key] > 3:
                            dense_samples += 1
                            continue
                        # traced program is the buggy program with the pysnooper decorator
                        traced_program = fail_info['program']
                        failed_test = extract_failed_test(traced_program, funcname)
                        if failed_test is None:
                            no_failed_test += 1
                            continue
                        # log is the terminal output of the traced program'
                        # log = fail_info['log']
                        # we do not need log for now
                        # trace is the trace of the traced program
                        raw_trace = fail_info['trace']
                        func_trace = extract_trace(raw_trace, inference=args.inference)
                        if func_trace is None:
                            no_func_trace += 1
                            continue
                        inlined_trace = attach_trace_to_code(traced_program, func_trace, state_order_flag=True, ignore_intermediate_flag=True)
                        # capture errors happening in the middle of the execution
                        inlined_trace = attach_execution_exception(inlined_trace, func_trace)
                        # clean pysnooper decorator and other stuff
                        inlined_trace = clean_code(inlined_trace)
                        if inlined_trace.strip() == "":
                            inlined_trace = "# No code generated. Please generate again."
                        # write the sample
                        sample = {
                            "raw_index": raw_index,
                            "nl": nl,
                            "solution": solution,
                            "buggy_trace": inlined_trace,
                            "failed_test": failed_test
                        }
                        sample_per_example[key] += 1
                        f_out.write(json.dumps(sample) + '\n')
                        total_samples += 1

    print(f"Total samples: {total_samples}")
    print(f"No expected output: {no_failed_test}")
    print(f"No function trace: {no_func_trace}")
    print(f"Dense samples: {dense_samples}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['prepare_data', 'construct_input'])
    # prepare_data
    parser.add_argument('--bug_report_file', type=str)
    parser.add_argument('--inference', action='store_true')

    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()

    prepare_data()

    