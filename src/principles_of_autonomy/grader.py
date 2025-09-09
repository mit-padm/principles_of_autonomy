# From https://github.com/RussTedrake/underactuated/blob/master/underactuated/exercises/grader.py

import json
import os
import glob
from pathlib import Path
import shutil
import sys
import re
import unittest
from runpy import run_path

from collections.abc import Iterable
import nbformat
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner
from nbconvert import PythonExporter, preprocessors
import zipfile

grader_throws = False

def set_grader_throws(val):
    global grader_throws
    grader_throws = val

class RemoveMagicCommands(preprocessors.Preprocessor):
    def preprocess_cell(self, cell, resources, index):
        if cell.cell_type == 'code':
            # Remove lines that start with % or %%
            cell.source = "\n".join(
                line for line in cell.source.splitlines()
                if not re.match(r"^\s*%{1,2}", line)
            )
        return cell, resources
 
class RemoveRawCells(preprocessors.Preprocessor):
    # raw cells don't error if run in notebook, but error when copied to py file
    def preprocess_cell(self, cell, resources, index):
        if cell.cell_type == 'raw':
            # remove all contents
            cell.source = ""
        return cell, resources

def get_locals(nb_locals, names: list) -> list:
    """
    Get variables from local and raise an error if they're not present 
    """
    out = []
    for name in names:
        try:
            out.append(nb_locals[name])
        except KeyError:
            raise RuntimeError(f"Test requires a variable/function with the name {name} to be defined. Make sure you have not changed the name of this variable, and that the cell that defines it in your notebook has been run without producing errors.")
    if len(names) == 1:
        return out[0]
    return out

def compare_iterators(ans, candidate) -> bool:
    if len(ans) != len(candidate):
        raise RuntimeError("Submitted answer is of incorrect length")
    for a, b in zip(ans, candidate):
        if isinstance(a, Iterable) and not isinstance(a, str):
            if not compare_iterators(a, b):
                return False
        else:
            if a != b:
                return False
    return True
class Grader:
    def __init__(self):
        pass

    @staticmethod
    def grade_from_notebooks(test_cases_list, notebook_ipynb_list, results_json):
        """
        Running notebooks in notebook_ipynb_list and evaluating
        them on test_cases_list. Result is written into results_json
        """
        try:
            notebook_locals_list = []
            for notebook_ipynb in notebook_ipynb_list:
                notebook_locals_list.append(Grader.locals_from_notebook(notebook_ipynb))
        except Exception as e:
            Grader.global_fail_with_error_message(
                "Exception when running file: " + notebook_ipynb + ", " + str(e),
                results_json,
            )
            raise

        # Grade notebook_locals_list on test_cases_list
        Grader.grade_output(test_cases_list, notebook_locals_list, results_json)

    @staticmethod
    def grade_output(test_case_list, notebook_locals_list, results_json):
        """Grading the notebook_locals with the provided test_cases"""
        # setup test suite for gradescope
        suite = unittest.TestSuite()
        for test_case, notebook_locals in zip(test_case_list, notebook_locals_list):
            test_case_names = unittest.defaultTestLoader.getTestCaseNames(test_case)
            for test_case_name in test_case_names:
                suite.addTest(test_case(test_case_name, notebook_locals))

        # run all the tests in the suite
        with open(results_json, "w") as fh:
            JSONTestRunner(stream=fh).run(suite)

    def run_single_test_inline(test_case, test_name, locals):
        suite = unittest.TestSuite()
        suite.addTest(test_case(test_name, locals))
        runner = unittest.TextTestRunner()
        runner.run(suite)

    @staticmethod
    def locals_from_notebook(notebook_ipynb):
        """Read, run, return locals of notebook"""
        banned_commands = ["HTML"]

        # temporary fix for deepnote weirdness
        # if you manually set the cwd of the notebook, it places it in a nested folder
        # waiting to hear back from them on how to address having to set a manual directory
        nb_path = Path(notebook_ipynb)
        if not nb_path.exists() and (nb_path.parent / "work" / nb_path.name).exists():
            notebook_ipynb = nb_path.parent / "work" / nb_path.name
        if not Path(notebook_ipynb).exists():
            raise RuntimeError(f"No notebook found: {Path(notebook_ipynb).name}. Make sure the notebook is in the submitted files and not in an odd directory (should be root or in /work)")


        ipynb = json.load(open(notebook_ipynb))

        for cell in ipynb["cells"]:
            # make sure cells are lists. Sometimes they end up being a string with all lines in them
            if type(cell['source']) is str:
                # split into lines
                cell['source'] = cell['source'].splitlines(keepends=True)

            # erase test cells, this is optional and useful for debugging
            # to avoid recursions when developing
            if any("## TEST ##" in line for line in cell["source"]):
                cell["source"] = []
            # filter out all the lines with banned commands
            if banned_commands is not None:
                cell["source"] = [
                    line
                    for line in cell["source"]
                    if not any(command in line for command in banned_commands)
                ]

        exporter = PythonExporter()
        exporter.register_preprocessor(RemoveMagicCommands, enabled=True)
        exporter.register_preprocessor(RemoveRawCells, enabled=True)
        source, meta = exporter.from_notebook_node(
            nbformat.reads(json.dumps(ipynb), nbformat.NO_CONVERT)
        )
        testing_dir = Path("/file_testing_dir")
        if testing_dir.exists():
            shutil.rmtree(str(testing_dir))
        # copy over files
        shutil.copytree("/autograder/submission/", str(testing_dir))
        student_converted_notebook_name = "student_cleaned_notebook.py"
        if (testing_dir / student_converted_notebook_name).exists():
            raise RuntimeError("Someone submitted a file named `student_cleaned_notebook.py`. This conflicts with the autograder. Please resubmit without the conflicting file.")
        with open(testing_dir / student_converted_notebook_name, "w") as fh:
            fh.write(source)
        sys.path.insert(0,str(testing_dir))
        cwd = os.getcwd()
        os.chdir(testing_dir)
        notebook_locals = run_path(testing_dir / student_converted_notebook_name)
        os.chdir(cwd)
        sys.path.pop(0)
        shutil.rmtree(str(testing_dir))
        return notebook_locals

    @staticmethod
    def global_fail_with_error_message(msg, results_json):
        """Error message if no specific"""
        results = {"score": 0.0, "output": msg}

        with open(results_json, "w") as f:
            f.write(
                json.dumps(
                    results,
                    indent=4,
                    sort_keys=True,
                    separators=(",", ": "),
                    ensure_ascii=True,
                )
            )

    @staticmethod
    def print_test_results(results_json):
        """Printing the results.json"""
        # open the json file for reading
        with open(results_json, "r") as fh:
            result = json.load(fh)

        # print total score
        max_score = sum(test["max_score"] for test in result["tests"])
        print("Total score is {}/{}.".format(int(result["score"]), max_score))

        if grader_throws and int(result["score"]) != max_score:
            raise RuntimeError("Grader did not award full points.")

        # print partial scores
        for test in result["tests"]:
            print(
                "\nScore for {} is {}/{}.".format(
                    test["name"], int(test["score"]), test["max_score"]
                )
            )

            # print error message if any
            if "output" in test:
                print("- " + test["output"])

    @staticmethod
    def prepare_submission(notebook_name):
        # Determine file system format
        input_nb = f"{notebook_name}.ipynb"
        file_match = glob.glob(f"**/*{input_nb}", recursive=True)
        if not file_match:
            raise NameError("Required files not found - Please submit homework manually.")
        active_folder = os.path.dirname(file_match[0])

        # Load input notebook
        input_nb = os.path.join(active_folder, input_nb)
        output_nb = os.path.join(active_folder, f"{notebook_name}_responses_only.ipynb")
        with open(input_nb, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Create results-only notebook
        new_nb = nbformat.v4.new_notebook()
        new_nb.cells = []

        # Extract "Write your answer in the cell below this one" + following cell
        for i, cell in enumerate(nb.cells):
            cell = nb.cells[i]
            if cell.cell_type == "markdown" and "Write your answer in the cell below this one." in cell.source:
                new_nb.cells.append(cell)
                if i + 1 < len(nb.cells):
                    new_nb.cells.append(nb.cells[i + 1])

        # Save results-only notebook
        with open(output_nb, "w", encoding="utf-8") as f:
            nbformat.write(new_nb, f)

        print(f"Open-ended responses written to {output_nb}")

        if not active_folder:
            active_folder = os.path.join("../", os.path.basename(os.getcwd()))

        folder_name = os.path.basename(active_folder)
        zip_name = f"{active_folder}/{folder_name}.zip"

        with zipfile.ZipFile(zip_name, "w") as zf:
            for file in os.listdir(active_folder):
                if os.path.isfile(os.path.join(active_folder, file)) and not file.endswith(".zip") :  # only add files, skip subfolders
                    zf.write(os.path.join(active_folder, file), arcname=file)  # arcname strips the folder name        zip_name = f"{os.path.basename(os.getcwd())}.zip"

        print(f"Compressed files in folder to {zip_name} (submit this to Gradescope)")


