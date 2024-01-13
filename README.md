# AI Based Road Inspection System

<p align="center">
  <img width="60%" height="60%" src="reports/omdena bhopal mexico.jpg">
</p>

## The Problem
Current practices of performing road inspections are time-consuming and labour-intensive. Road surfaces degrade on a daily basis as a result of the heavy traffic on them. This will not only impact the driver’s comfort but will also impact economic efficiency. To maintain roads as efficiently as possible, municipalities perform regular inspections. The aim of the project is to use machine learning to study and analyze different types of road defects and to automatically detect any road abnormalities. We will design, build and test an inspection system for this purpose. The system is equipped with a camera to collect video streams from different roads with and without defects. Then, the captured data will be analyzed using the Matlab machine learning toolbox to train and test the network. Finally, the system will provide recommended actions for the municipality related to actions required to fix/correct road defects. The approach is divided into 3 main tasks: Data acquisition, Data Training/Testing, and Dashboard Building and Testing.

## The Project Goals
The goal of this project is to design, build and test an inspection system for detecting road abnormalities, defects, and damages using machine learning. The proposed system aims to improve the efficiency of road inspections and reduce the time and labor required for the process. The system will be equipped with a camera to capture video streams from different roads, and the data will be analyzed using the Matlab machine learning toolbox to train and test the network. The output of the system will be recommended actions for the municipality to fix/correct any identified road defects. The approach will involve three main tasks: data acquisition, data training/testing, and dashboard building and testing. Ultimately, the proposed system will help to maintain roads more efficiently, enhance driver comfort, and improve economic efficiency. Additionally, the system will provide insights into the causes of road abnormalities in Indian roads, including pitfalls, sinks, flooding, and traffic congestion due to insufficient lanes in cities and towns.

## Contribution Guidelines
- Have a Look at the [project structure](#project-structure) and [folder overview](#folder-overview) below to understand where to store/upload your contribution
- If you're creating a task, Go to the task folder and create a new folder with the below naming convention and add a README.md with task details and goals to help other contributors understand
    - Task Folder Naming Convention : _task-n-taskname.(n is the task number)_  ex: task-1-data-analysis, task-2-model-deployment etc.
    - Create a README.md with a table containing information table about all contributions for the task.
- If you're contributing for a task, please make sure to store in relavant location and update the README.md information table with your contribution details.
- Make sure your File names(jupyter notebooks, python files, data sheet file names etc) has proper naming to help others in easily identifing them.
- Please restrict yourself from creating unnessesary folders other than in 'tasks' folder (as above mentioned naming convention) to avoid confusion. 

## Project Structure

    ├── LICENSE
    ├── README.md          <- The top-level README for developers/collaborators using this project.
    ├── original           <- Original Source Code of the challenge hosted by omdena. Can be used as a reference code for the current project goal.
    │ 
    │
    ├── reports            <- Folder containing the final reports/results of this project
    │   └── README.md      <- Details about final reports and analysis
    │ 
    │   
    ├── src                <- Source code folder for this project
        │
        ├── data           <- Datasets used and collected for this project
        │   
        ├── docs           <- Folder for Task documentations, Meeting Presentations and task Workflow Documents and Diagrams.
        │
        ├── references     <- Data dictionaries, manuals, and all other explanatory references used 
        │
        ├── tasks          <- Master folder for all individual task folders
        │
        ├── visualizations <- Code and Visualization dashboards generated for the project
        │
        └── results        <- Folder to store Final analysis and modelling results and code.
--------

## Folder Overview

- Original          - Folder Containing old/completed Omdena challenge code.
- Reports           - Folder to store all Final Reports of this project
- Data              - Folder to Store all the data collected and used for this project 
- Docs              - Folder for Task documentations, Meeting Presentations and task Workflow Documents and Diagrams.
- References        - Folder to store any referneced code/research papers and other useful documents used for this project
- Tasks             - Master folder for all tasks
  - All Task Folder names should follow specific naming convention
  - All Task folder names should be in chronologial order (from 1 to n)
  - All Task folders should have a README.md file with task Details and task goals along with an info table containing all code/notebook files with their links and information
  - Update the [task-table](./src/tasks/README.md#task-table) whenever a task is created and explain the purpose and goals of the task to others.
- Visualization     - Folder to store dashboards, analysis and visualization reports
- Results           - Folder to store final analysis modelling results for the project.


