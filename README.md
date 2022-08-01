# Report Manager
*Version: 0.0.1*

Manage your reports
- Merge reports based off specified columns
- Generate analysis on data set

### Usage:
| Short Flag | Long Flag | Description    |
| --- | ------|---------------------------|
| -h | --help | See Usage                 |
| -f | --file | File                      |

### Example:
```bash
report-manager 
```


#### Build Instructions
Build Python Package

```bash
sudo chmod +x ./*.py
sudo pip install .
python3 setup.py bdist_wheel --universal
# Test Pypi
twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose -u "Username" -p "Password"
# Prod Pypi
twine upload dist/* --verbose -u "Username" -p "Password"
```
