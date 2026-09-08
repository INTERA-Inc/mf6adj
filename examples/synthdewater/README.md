# Synthetic dewatering model

A MODFLOW 6 model of a synthetic dewatering problem, used by
[`dewater_demo.ipynb`](../dewater_demo.ipynb). The notebook copies this directory to
`synthdewater_working` and runs it there, so nothing here is modified.

The model, and the history matching and optimization study it came from, were built by
`buildmodel_workflow.py.txt`. That script used pyemu and PEST++, was never executed by the
notebooks or the tests, and was removed when mf6adj stopped depending on pyemu. To read it:

```shell
git log --diff-filter=D --format=%H -1 -- examples/synthdewater/buildmodel_workflow.py.txt
git show <commit>^:examples/synthdewater/buildmodel_workflow.py.txt
```
