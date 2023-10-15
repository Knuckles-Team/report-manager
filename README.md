# Report Manager

![PyPI - Version](https://img.shields.io/pypi/v/report-manager)
![PyPI - Downloads](https://img.shields.io/pypi/dd/report-manager)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/report-manager)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/report-manager)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/report-manager)
![PyPI - License](https://img.shields.io/pypi/l/report-manager)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/report-manager)

![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/report-manager)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/report-manager)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/report-manager)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/report-manager)

![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/report-manager)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/report-manager)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/report-manager)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/report-manager)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/report-manager)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/report-manager)


*Version: 0.10.0*

Manage your reports
- Merge reports based off specified columns
- Generate analysis on data set
- Pandas profiling

This repository is actively maintained - Contributions are welcome!

<details>
  <summary><b>Usage:</b></summary>

| Short Flag | Long Flag          | Description                                                                    |
|------------|--------------------|--------------------------------------------------------------------------------|
| -h         | --help             | See Usage                                                                      |
| -f         | --files            | File(s) to be read (Comma separated)                                           |
| -n         | --name             | Name of report                                                                 |
| -j         | --join-keys        | File(s) to be read (Pipe Separated for files, Comma separated for each column) |
| -t         | --type             | Save as the following formats: <CSV/csv/XLSX/xlsx>                             |
| -m         | --merge            | Merge two datasets: <inner/outer/left/right/append>                            |
| -p         | --pandas-profiling | Generate a pandas profiling report                                             |
| -r         | --report           | Generate a custom report with plots                                            |

</details>

<details>
  <summary><b>Example:</b></summary>

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

</details>

<details>
  <summary><b>Installation Instructions:</b></summary>

Install Python Package

```bash
python -m pip install report-manager
```

</details>

<details>
  <summary><b>Repository Owners:</b></summary>


<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)
</details>
