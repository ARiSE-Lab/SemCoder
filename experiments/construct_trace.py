"""
Utility functions to construct trace information from the trace log.
"""

import json
import re

def handle_value(var_value):
    """
    Handle value as json object in the trace log.
    """
    # str does not need to be handled
    if var_value.startswith("[") and var_value.endswith("]"):
        var_value = var_value.replace("'", '"')
    try:
        var_value = json.loads(var_value)
        return var_value
    except:
        if var_value == "None":
            return None
        elif var_value == "''":
            return ""
        elif var_value == "[]":
            return []
        elif var_value == "{}":
            return {}
        elif var_value == "True":
            return True
        elif var_value == "False":
            return False
        elif var_value.startswith("'"): # seems pysnooper only gives single quote for string?
            return var_value[1:-1]
        else:
            return var_value

def convertv2s(obj, output=False):
    """
    Convert value to string.
    """
    # Handle boolean
    if isinstance(obj, bool):
        return str(obj)
    # Handle None
    elif obj is None:
        return "None"
    # Handle string
    elif isinstance(obj, str):
        if output:
            # output value is standalone, rather than json object, so we need to add double quotes
            if obj.startswith('"') and obj.endswith('"') or \
            obj.startswith("'") and obj.endswith("'") or \
            obj.startswith("(") and obj.endswith(")") or \
            obj.startswith("{") and obj.endswith("}"):
                return obj
            else:
                return f'"{obj}"'
        else:
            return obj
    # Handle other types
    else:
        return json.dumps(obj)

def concise_spad(line: str) -> str:
    """
    Remove intermediate states (3 to n-1) of the spad trace: replace them with '...'
    Only <= 3 states are kept.
    """
    # count how many [STATE] are there in the line
    state_count = line.count("[STATE")
    if state_count <= 3:
        return line
    # locate the third [STATE and the last [STATE
    third_state = line.find("[STATE", line.find("[STATE", line.find("[STATE") + 1) + 1)
    last_state = line.rfind("[STATE")
    assert third_state != -1 and last_state != -1, f"Cannot find the third and the last [STATE] in the line: {line}"
    # replace the states between the third and the last with '...'
    line = line[:third_state] + " ... " + line[last_state:]

    return line

def attach_trace_to_code(source_code, trace_log, state_order_flag, ignore_intermediate_flag):
    """
    Attach trace information to the source code as spad style.
    """
    # Define patterns to identify trace lines and variable assignments
    var_pattern = re.compile(r"^(New var|Modified var):\.* (.*?) = (.*)")
    input_pattern = re.compile(r"Starting var:\.* (.*?) = (.*)")
    output_pattern = re.compile(r"Return value:\.* (.*)")
    
    trace_info = {}
    inputs = {}
    outputs = []
    first_line = -1 # first line will be attached with the input state
    currnet_line = 0 # keep track of the current source line number corresponding to the trace
    line2times = {} # how many times a line is executed (0-indexed)
    global_state_order = 0 # keep track of the global order of state changes

    # Parse the trace log
    state_change = False
    for line in trace_log.split('\n'):
        line = line.strip()
        if input_match := input_pattern.match(line):
            k, v = input_match.group(1), input_match.group(2)
            inputs[k] = handle_value(v)
        elif output_match := output_pattern.match(line):
            v = output_match.group(1)
            outputs.append(handle_value(v))
        elif var_match := var_pattern.match(line):
            var_name, var_value = var_match.group(2), handle_value(var_match.group(3))
            trace_info.setdefault(currnet_line, []).append((line2times[currnet_line], global_state_order, var_name, var_value))
            state_change = True
        else:
            line_split = line.split()
            if len(line_split) < 3:
                continue
            line_number = line_split[2]
            # make sure the line number is an integer, otherwise continue
            try:
                line_number = int(line_number)
            except:
                continue
            currnet_line = line_number
            if first_line == -1:
                first_line = currnet_line
            # record how many times a line is executed
            if currnet_line not in line2times:
                line2times[currnet_line] = -1 # for 0-based index
            line2times[currnet_line] += 1 
            if state_change: # we only maintain the relative order of state changes
                global_state_order += 1
                state_change = False
    
    # Add output to the last line of the function
    if outputs:
        last_line_number = currnet_line
        if last_line_number not in trace_info:
            trace_info[last_line_number] = []
        trace_info[last_line_number].append((0, global_state_order, "[OUTPUT]", outputs[-1]))

    # Process the source code
    modified_lines = []
    source_lines = source_code.split('\n')
    for i, line in enumerate(source_lines, 1):
        modified_line = line
        # add input to the first line of the function
        if i == first_line:
            inputs_str = ' # [INPUT] ' + convertv2s(inputs) + ' [/INPUT]'
            modified_line += inputs_str
        if i in trace_info:
            states = [] # state of this line during different times of execution
            for var_state in trace_info[i]:
                l2t, state_order, var_name, var_value = var_state
                # check if the state belongs to the current line
                if len(states) > l2t: # if the state is already recorded
                    states[l2t][1][var_name] = var_value
                else:
                    states.append((state_order, {var_name: var_value}))
            if len(states) > 0:
                modified_line += ' # '
                for s in states:
                    order, state = s
                    if state_order_flag:
                        if "[OUTPUT]" in state:
                            state_str = '[OUTPUT] ' + convertv2s(state["[OUTPUT]"], output=True) + ' [/OUTPUT]'
                            state.pop("[OUTPUT]")
                            # if some other state is also present, add the output to the end of the line
                            if state:
                                state_str = f'[STATE-{order}] ' + json.dumps(state) + f' [/STATE-{order}]' + state_str
                        else:
                            state_str = f'[STATE-{order}] ' + json.dumps(state) + f' [/STATE-{order}]'
                    else:
                        if "[OUTPUT]" in state:
                            state_str = '[OUTPUT] ' + convertv2s(state["[OUTPUT]"], output=True) + ' [/OUTPUT]'
                            state.pop("[OUTPUT]")
                            # if some other state is also present, add the output to the end of the line
                            if state:
                                state_str = '[STATE] ' + json.dumps(state) + ' [/STATE]' + state_str
                        else:
                            state_str = '[STATE] ' + json.dumps(state) + ' [/STATE]'
                    modified_line += state_str
        if ignore_intermediate_flag:
            modified_line = concise_spad(modified_line)
        modified_lines.append(modified_line)

    # Join modified lines back into a single string
    return '\n'.join(modified_lines)
