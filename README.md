# Report Manager
*Version: 0.9.0*

Manage your reports
- Merge reports based off specified columns
- Generate analysis on data set
- Pandas profiling

### Usage:
| Short Flag | Long Flag | Description                                         |
|------------| ------|-----------------------------------------------------|
| -h         | --help | See Usage                                           |
| -f         | --files | File(s) to be read (Comma separated)     |
| -n         | --name | Name of report                                      |
| -j         | --join-keys  | File(s) to be read (Pipe Separated for files, Comma separated for each column)     |
| -t         | --type | Save as the following formats: <CSV/csv/XLSX/xlsx>  |
| -m         | --merge | Merge two datasets: <inner/outer/left/right/append> |
| -p         | --pandas-profiling | Generate a pandas profiling report                  |
| -r         | --report | Generate a custom report with plots                 |


### Example:

Report and Pandas Profiling
```bash
report-manager --pandas-profiling --report
          --files "/home/Users/Fred/usa_weather.csv" 
          --name "USA Weather" 
          --type "XLSX" 
          --save-directory "/home/Users/Fred/Downloads"
```

Merge
```bash
report-manager --merge "append"
          --files "/home/Users/Fred/usa_weather.csv,/home/Users/Fred/mexico_weather.csv" 
          --name "North America Weather" 
          --type "csv" 
          --save-directory "/home/Users/Fred/Downloads" 
          --join-keys "column1,column2,column3"
```

#### Install Instructions
Install Python Package

```bash
python -m pip install report-manager
```

#### Build Instructions
Build Python Package

```bash
sudo chmod +x ./*.py
sudo pip install .
python setup.py bdist_wheel --universal
# Test Pypi
twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose -u "Username" -p "Password"
# Prod Pypi
twine upload dist/* --verbose -u "Username" -p "Password"
```
