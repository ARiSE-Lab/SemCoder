from typing import List, Dict, Tuple, Any
import os, sys, json, ast, time, io, tokenize
import evalplus.data.humaneval
from thefuzz import fuzz
import numpy as np

import evalplus.data
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.data.mbpp import mbpp_deserialize_inputs

import sys, io, queue, psutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import argparse

class Shell:
    def __init__(
        self,
        shell_exec: bool = True,
        print_out: bool = True,
        print_cmd: bool = True,
        print_file: io.TextIOWrapper | None = None,
        print_stderr_file: io.TextIOWrapper | None = None,
        return_list: bool = False,
    ) -> None:
        self.shell_exec = shell_exec
        self.print_out = print_out
        self.print_cmd = print_cmd
        self.print_file = print_file
        if print_stderr_file is None and print_file is not None:
            self.print_stderr_file = print_file
        else:
            self.print_stderr_file = print_stderr_file
        self.return_list = return_list


    def _read_popen_pipes(self, p: subprocess.Popen, timeout_sec: float|None = None):

        def _enqueue_output(file: io.TextIOWrapper, q: queue.Queue):
            for line in iter(file.readline, ''):
                q.put(line)
            file.close()

        def _timeout():
            try:
                p.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                parent = psutil.Process(p.pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()

        with ThreadPoolExecutor(3) as pool:
            q_stdout, q_stderr = queue.Queue(), queue.Queue()

            if timeout_sec is not None:
                pool.submit(_timeout)
            pool.submit(_enqueue_output, p.stdout, q_stdout)
            pool.submit(_enqueue_output, p.stderr, q_stderr)

            while p.poll() is None or not q_stdout.empty() or not q_stderr.empty():
                out_line = err_line = ''

                try:
                    out_line = q_stdout.get_nowait()
                except queue.Empty:
                    pass

                try:
                    err_line = q_stderr.get_nowait()
                except queue.Empty:
                    pass

                yield (out_line, err_line)
    

    def run(self, cmd: str | List[str], timeout: float|None = None) -> Tuple[str|List[str], str|List[str], int]:
        with subprocess.Popen(
            cmd, shell=self.shell_exec, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ) as p:
            if self.print_cmd:
                if self.print_out:
                    print(f'+ {cmd}', file=sys.stderr, flush=True)
                if self.print_file:
                    print(f'+ {cmd}', file=self.print_file, flush=True)
            out: List[str] = []
            err: List[str] = []
            for out_line, err_line in self._read_popen_pipes(p, timeout):
                out.append(out_line)
                err.append(err_line)
                if self.print_out:
                    print(out_line, end='', flush=True)
                    print(err_line, end='', file=sys.stderr, flush=True)
                if self.print_file:
                    print(out_line, end='', flush=True, file=self.print_file)
                if self.print_stderr_file:
                    print(err_line, end='', flush=True, file=self.print_stderr_file)
            # end for
            if self.return_list:
                return out, err, p.returncode
            else:
                return ''.join(out), ''.join(err), p.returncode
    

    def bash(self, cmd: str, timeout: float|None = None) -> Tuple[str, str, int]:
        _prev_shell_exec = self.shell_exec
        self.shell_exec = True
        assert "'" not in cmd, f'cmd should not contain single quote: {cmd}'
        ret = self.run(f'bash -c \'{cmd}\'', timeout)
        self.shell_exec = _prev_shell_exec
        return ret


def remove_comments_and_docstrings(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    # out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out

'''
These can trigger bugs of remove_comments_and_docstrings:

```python
import re
from collections import Counter

# List of common English stopwords
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
"you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 
'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 
'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 
'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 
'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 
'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', 
"mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def top_n_common_words(text, n):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text into words and convert to lower case
    words = text.lower().split()
    
    # Remove stopwords
    words = [word for word in words if word not in stopwords]
    
    # Count the frequency of each word
    word_counts = Counter(words)
    
    # Return the top n most common words along with their frequencies
    return word_counts.most_common(n)
```

s = "\ndef is_woodall(x): \n\tif not isinstance(x, int):\n\t\treturn False\n\tif x <= 0 or x % 2 == 0:\n\t\treturn False\n\tif (x == 1): \n\t\treturn True\n\tx += 1 \n\ti = 0\n\twhile (x % 2 == 0): \n\t\tx /= 2\n\t\ti += 1\n\t\tif (i == x): \n\t\t\treturn True\n\treturn False\n"
'''


def normalize_code(code: str) -> str:
    code = ast.unparse(ast.parse(code))
    code = remove_comments_and_docstrings(code)
    code = ast.unparse(ast.parse(code))
    return code


def gen_entrypoint(funcname: str, io_dict: Dict[str, str]) -> str|None:
    ep = f'{funcname}('
    ep += ', '.join([
        repr(a) for a in eval(io_dict['args'])
    ])
    ep += ', '
    ep += ', '.join([
        f'{k}={repr(v)}' for k, v in eval(io_dict['kwargs']).items()
    ])
    ep += ')'
    return ep


def detect_first_codeblock(text: str) -> str:
    lines = text.splitlines(keepends=True)
    codeblocks: list[str] = []
    start_index: int | None = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("```"):
            if start_index is None:
                start_index = idx
            else:
                codeblocks.append("".join(lines[start_index + 1 : idx]))
                start_index = None
                break
    return '\n'.join(codeblocks).strip()


def get_funcname_linepos(code: str, codelines: List[str]) -> Tuple[str, int, int]:
    '''Only return the function name and its line position in the code when there is only one function.'''
    try:
        tree = ast.parse(code)
    except:
        return '', -1, -1
    blacklist = [ast.ClassDef]
    func_node = None
    for node in tree.body:
        if any(isinstance(node, bl) for bl in blacklist):
            return '', -1, -1
        if isinstance(node, ast.FunctionDef):
            if func_node:
                return '', -1, -1
            func_node = node
    if not func_node:
        return '', -1, -1
    return func_node.name, func_node.lineno - 1, func_node.end_lineno


def safety_check(
    funcname: str, func: str, ios: Dict[str, str], rank: int,
) -> bool:
    
    def scan_strace_log(strace_path: str) -> str:
        with open(strace_path) as f:
            for line in f:
                if (
                    'pycache' not in line and
                    any(s in line for s in [
                        'mkdir', 'unlink', 'chmod', 'chown', 'rmdir', 'rename',
                        'O_WRONLY', 'O_RDWR',
                    ])
                ):
                    return line
        return ''
    
    prog = f'''{func}\n\n\n'''
    for io_dict in ios:
        ep = gen_entrypoint(funcname, io_dict)
        prog += f'''try:
    {ep}
except:
    pass

'''
    tmpdir = f'{args.output_dir}/tmp_refine_exec/{rank}'
    os.makedirs(tmpdir, exist_ok=True)
    sol_filename = 'sol.py'
    sol_path = os.path.join(tmpdir, sol_filename)
    out_file = os.path.join(tmpdir, 'sol_out.log')
    with open(sol_path, 'w') as f:
        f.write(prog)
    strace_file = 'strace.log'
    strace_path = os.path.join(tmpdir, strace_file)
    os.remove(strace_path) if os.path.exists(strace_path) else None
    Shell(print_out=False).run(f'docker kill tmprun_{rank}; docker rm tmprun_{rank}')
    img = 'co1lin/ubuntu-basic'
    timeout = 20
    cmd = f'timeout -s 9 {timeout} docker run --rm --name tmprun_{rank}' \
          f' -v {os.path.abspath(tmpdir)}:/{tmpdir}:rw {img} ' \
          f'bash -c "set -e; source /home/ubuntu/miniforge3/bin/activate; ' \
          f'mkdir -p /home/ubuntu/tmprun; ' \
          f'cp /{tmpdir}/{sol_filename} /home/ubuntu/tmprun; ' \
          f'cd /home/ubuntu/tmprun; ' \
          f'strace -f -ttt -T -o {strace_file} sh -c \'python {sol_filename} < /dev/null\'; ' \
          f'cp /home/ubuntu/tmprun/{strace_file} /{tmpdir}"'
    with open(out_file, 'w') as f_sol_out:
        out, err, code = Shell(
            print_out=False,
            print_cmd=True,
            print_file=f_sol_out,
        ).run(cmd)
    Shell(print_out=False).run(f'docker kill tmprun_{rank}; docker rm tmprun_{rank}')
    if os.path.exists(strace_path):
        return scan_strace_log(strace_path)
    elif code == 128 + 9:
        return '' # timeout with kill -9
    # no strace log and not timeout
    return 'Program is killed due to unknown reason.'


def exec_code(code: str, rank: int, tmpdir: str, timeout: float = 10) -> Tuple[str, str, int]:
    os.makedirs(tmpdir, exist_ok=True)
    sol_filename = 'solution.py'
    sol_path = os.path.join(tmpdir, sol_filename)
    out_file = os.path.join(tmpdir, 'sol_out.log')
    with open(sol_path, 'w') as f:
        f.write(code)
    cmd = f'timeout -s 9 {timeout} sh -c "python {sol_path} < /dev/null"'
    with open(out_file, 'w') as f_sol_out:
        out, err, code = Shell(
            print_out=False,
            print_cmd=True,
            print_file=f_sol_out,
        ).run(cmd)
    return out, err, code


def extract_func_from_response(response: str) -> Tuple[str, str]:
    code = detect_first_codeblock(response)
    if not code:
        return '', ''
    codelines = code.splitlines()
    funcname, pos, endpos = get_funcname_linepos(code, codelines)
    if not funcname:
        return '', ''
    assert 0 <= pos < endpos <= len(codelines), f'{pos = }, {endpos = }, {len(codelines) = }'
    func = '\n'.join(codelines[:endpos])
    return funcname, func


def is_floats(x) -> bool:
    # check if it is float; List[float]; Tuple[float]
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)) and x:
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False


MBPP_OUTPUT_NOT_NONE_TASKS = ["check_str", "text_match_three", "text_starta_endb"]
MBPP_OUTPUT_SET_EQ_TASKS = [
    "similar_elements",  # Mbpp/2
    "find_char_long",  # Mbpp/7
    "common_in_nested_lists",  # Mbpp/111
    "extract_singly",  # Mbpp/140
    "larg_nnum",  # Mbpp/232
    "intersection_array",  # Mbpp/249
    "find_dissimilar",  # Mbpp/579
    "Diff",  # Mbpp/769
]
def gen_assertion(funcname: str, out: str, exp: str, atol: float, benchmark: str) -> str:

    if benchmark == "mbpp":
        if "are_equivalent" == funcname:  # Mbpp/164 special oracle
            return 'assert True'
        elif "sum_div" == funcname:  # Mbpp/295 special oracle
            return f'assert {out} == {exp} or {out} == 0'
        elif funcname in MBPP_OUTPUT_SET_EQ_TASKS:
            return f'assert set({out}) == set({exp})'
        elif funcname in MBPP_OUTPUT_NOT_NONE_TASKS:
            # exp is True  if not None
            #        False if None
            return f'''if isinstance({out}, bool):
        assert {out} == {exp}
    else:
        assert {exp} == ({out} is not None)'''
    
    if benchmark == "humaneval":
        if "find_zero" == funcname:
            return f'assert poly(*inp, {out}) <= {atol}'

    if atol == 0 and is_floats(eval(exp)):
        atol = 1e-6  # enforce atol for float comparison
    if atol != 0:
        # explicitly set rtol=1e-07
        # to match `np.testing.assert_allclose`'s default values
        return f'''assert type({out}) == type({exp})
    if isinstance({exp}, (list, tuple)):
        assert len({out}) == len({exp})
    assert np.allclose({out}, {exp}, rtol=1e-07, atol={atol})'''
    else:
        return f'assert {out} == {exp}'


def gen_failures(
    funcname: str, func: str, ios: Dict[str, str],
    atol: float, benchmark: str, rank: int,
) -> Dict[str, Any] | None:
    """Returns:
        failures = [
            {"args": ..., "kwargs": ..., "log": ..., "program": ..., },
        ]
    """
    tmpdir = f'{args.output_dir}/tmp_refine_exec/{rank}'
    os.makedirs(tmpdir, exist_ok=True)
    # exec & trace
    assert 'check_ans_7d1' not in func
    trace_path = os.path.join(tmpdir, 'trace.log')
    failures = []
    for io_dict in ios:
        ep = gen_entrypoint(funcname, io_dict)
        assertion = gen_assertion(
            funcname, 'retv', io_dict['retv'], atol, benchmark,
        )
        timeout = 10
        # special cases
        if benchmark == 'humaneval' and 'find_zero' == funcname:
            assertion = assertion.replace('*inp', repr(eval(io_dict['args'])[0]))
        if benchmark == 'mbpp' and 'amicable_numbers_sum' == funcname:
            timeout = 1
        
        prog = ''
        if 'np.allclose' in assertion:
            prog = f'import numpy as np\n'
        
        prog += f'''import pysnooper
{func}

@pysnooper.snoop("{trace_path}", color=False, depth=2)
def check_ans_7d1():
    retv = {ep}
    {assertion}

check_ans_7d1()'''
        try:
            ast.parse(prog)
        except:
            print(funcname, flush=True)
            print(prog, flush=True)
            print(io_dict, flush=True)
            raise
        os.remove(trace_path) if os.path.exists(trace_path) else None
        out, err, code = exec_code(prog, rank, tmpdir, timeout)
        if code != 0:
            trace_str = '' # no trace due to syntax error
            if os.path.exists(trace_path):
                with open(trace_path) as f:
                    trace_str = f.read()
            if code == 128 + 9:
                err = f'Program is killed due to timeout in {timeout} seconds.'
            failures.append({
                'args': io_dict['args'],
                'kwargs': io_dict['kwargs'],
                'log': err,
                'program': prog,
                'trace': trace_str,
            })

    return failures


def gen_mp(
    dps: List[dict], output_file: str, benchmark: str, rank: int,
) -> None:
    f_out = open(output_file, 'a')
    print(f'{rank = } | {len(dps)}', flush=True)
    s_time = time.time()
    for i, dp in enumerate(dps):

        if i % 20 == 0:
            duration = time.time() - s_time
            eta = duration / i * (len(dps) - i) // 60 if i > 0 else -1
            print(f'{rank} | {i} / {len(dps)} ({i/len(dps)*100:.0f}%) | {duration = :.0f} sec | {eta = } min', flush=True)

        failed_func = dp.pop('failed_response')
        failed_inputs = dp.pop('failed_inputs')

        # generate reference ios
        ios: List[dict] = []
        cano_func, funcname = dp['func'], dp['funcname']
        for failed_input in failed_inputs:
            ep = gen_entrypoint(funcname, failed_input)
            ep = f'retv_7d1 = {ep}'
            prog = f'''{cano_func}\n\n\n{ep}'''
            exec_globals = {}
            try:
                exec(prog, exec_globals)
            except:
                print(repr(prog), flush=True)
                print(prog, flush=True)
                raise
            retv = exec_globals['retv_7d1']
            if benchmark == 'mbpp' and funcname in MBPP_OUTPUT_NOT_NONE_TASKS:
                retv = retv is not None
            ios.append({
                'args': failed_input['args'],
                'kwargs': failed_input['kwargs'],
                'retv': repr(retv),
            })
        dp['ios'] = ios
        
        ref_func_norm = normalize_code(dp['func'])
        failed_func_norm = normalize_code(failed_func)

        buggy_response = {
            'response': failed_func,
            'funcname': funcname,
            'func': failed_func,
            'sim_to_ref': fuzz.token_sort_ratio(ref_func_norm, failed_func_norm),
        }
        # fio_log = safety_check(funcname, failed_func, dp['ios'], rank)
        fio_log = None
        if fio_log:
            if 'unknown reason' not in fio_log:
                fio_log = f'Program makes changes to the file system which is not allowed.\n{fio_log}'
            buggy_response['failures'] = [
                {
                    'args': ios[0]['args'],
                    'kwargs': ios[0]['kwargs'],
                    'program': failed_func,
                    'log': fio_log,
                    'trace': '',
                }
            ]
        else:
            if funcname == "find_zero":
                continue
            failures = gen_failures(
                funcname, failed_func, dp['ios'],
                dp['atol'], benchmark, rank,
            )
            if len(failures) == 0:
                print(funcname, flush=True)
                print(failed_func, flush=True)
                print(dp['ios'], flush=True)
            assert len(failures) > 0
            buggy_response['failures'] = failures
        dp = {
            "buggy_responses": [buggy_response],
            **dp,
        }
        
        f_out.write(json.dumps(dp) + '\n')
    # end for dp
    f_out.close()


def gen(eval_result_file: str, output_file: str, version: str = 'base', evalplus_file: str = ''):
    import multiprocessing as mp
    NUM_PROC = 8
    dps = []
    # read eval_results and evalplus benchmark
    evalplus_data = {}
    if 'humaneval' in eval_result_file.lower():
        benchmark = 'humaneval'
        if evalplus_file:
            evalplus.data.humaneval.HUMANEVAL_OVERRIDE_PATH = evalplus_file
        evalplus_data = get_human_eval_plus()
    elif 'mbpp' in eval_result_file.lower():
        benchmark = 'mbpp'
        if evalplus_file:
            evalplus.data.mbpp.MBPP_OVERRIDE_PATH = evalplus_file
        evalplus_data = get_mbpp_plus()
    else:
        raise ValueError(f'Unknown evalplus_file: {eval_result_file}')
    
    with open(eval_result_file) as f:
        eval_results = json.load(f)['eval']

    # merge eval_results and evalplus
    for task_id, task_res in eval_results.items():
        task_res = task_res[0]
        if version == 'base' and task_res['base_status'] == 'pass':
            continue
        elif version == 'plus' and task_res['base_status'] == task_res['plus_status'] == 'pass':
            continue
        evalplus_task = evalplus_data[task_id]
        # remove docstrings in prompt (for humaneval) to avoid redundant instruction
        func = normalize_code(
            evalplus_task['prompt'] + evalplus_task['canonical_solution']
        )
        failed_inputs = task_res['base_fail_tests']
        if version == 'plus':
            failed_inputs += task_res['plus_fail_tests']
        if failed_inputs and benchmark == 'mbpp':
            failed_inputs = mbpp_deserialize_inputs(task_id, failed_inputs)
        if not failed_inputs:
            failed_inputs = evalplus_task['base_input'][:1]
        failed_inputs = [
            {
                'args': repr(args),
                'kwargs': repr({}),
            }
            for args in failed_inputs
        ]
        # assert task_res['solution'] is syntax correct after sanitization
        failed_response = normalize_code(task_res['solution'])
        dp = {
            'raw_index': task_id,
            'instruction': evalplus_task['prompt'],
            'funcname': evalplus_task['entry_point'],
            'func': func,
            'failed_response': failed_response,
            'failed_inputs': failed_inputs[:1],
            'atol': evalplus_task['atol'],
        }
        dps.append(dp)
    # end for
    print(f'num dps: {len(dps)}', flush=True)
    chunk_size = len(dps) // NUM_PROC + 1
    chunks = [dps[i:i + chunk_size] for i in range(0, len(dps), chunk_size)]
    rets = []
    with mp.Pool(NUM_PROC) as pool:
        for i, chunk in enumerate(chunks):
            ret = pool.apply_async(
                gen_mp, args=(chunk, output_file, benchmark, i),
            )
            rets.append(ret)
        pool.close()
        pool.join()
    for ret in rets:
        ret.get()
    print(f'\n\n========\nfinished all!', flush=True)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--eval_result_file', type=str, required=True)
    argparser.add_argument('--output_file', type=str, required=True)
    argparser.add_argument('--version', type=str, default='plus')
    argparser.add_argument('--evalplus_file', type=str, default=None)

    args = argparser.parse_args()
    args.output_dir = os.path.dirname(args.output_file)
    eval_result_file = args.eval_result_file
    output_file = args.output_file
    version = args.version
    evalplus_file = args.evalplus_file

    gen(eval_result_file, output_file, version, evalplus_file)
