# Installation
Please follow the guide below to install RegArch and its required dependent software. Before you begin, ensure that your system meets the necessary requirements.

## System requirements
RegArch has been developed and tested on Windows and macOS operating systems. While it might work on other systems such as Linux, it hasn't been fully tested on these platforms, so use caution when installing on non-tested systems.

## Python requirements
RegArch is developed and tested using Python 3.11. It is recommended that you use this version or a compatible one to ensure optimal compatibility and performance.


## Installation from source 
To ensure a clean installation that doesn't interfere with other Python projects on your system, it is recommended to create a new Python virtual environment specifically for RegArch and install the required libraries within it. Follow these steps:

### Step 1: Clone the Repository
First, clone the RegArch repository to your local machine:
```shell
$ git clone https://github.com/yuxiaochen11/RegArch.git
```

### Step 2: Create a Virtual Environment
Navigate to the cloned repository directory and create a new Python virtual environment. You can replace ```env``` with your desired environment name:

```shell
$ cd RegArch
$ python -m venv env
```

### Step 3: Activate the Virtual Environment
Activate the newly created virtual environment to ensure subsequent Python commands are executed within this environment:

* On Windows
    ```shell
    $ .\env\Scripts\activate
    ```


* On macOS or Linux
    ```shell
    $ source env/bin/activate
    ```  
After activation, the command prompt will typically show the environment name (for example (```env```)) indicating that you are currently working within the virtual environment.

### Step 4: Install RegArch
With the virtual environment activated, run the following command to install RegArch and its dependencies:
```shell
(env) $ python -m pip install -e ".[demo]"
```
Once the installation is complete, RegArch and all required libraries will be installed in your virtual environment. You can now import and use RegArch in your project development within this environment.


### Additional Tips

* __Update Build Tools__: While our installation method doesn't require pip directly, it's beneficial to ensure that build tools like setuptools and wheel are up to date, as they assist in the installation process. If you have pip available, you can update these tools within the virtual environment:

```shell
(env) $ python -m pip install --upgrade pip setuptools wheel
```
This step is optional but can help prevent potential build or compatibility issues.

*__Deactivate the Environment__: Once installation is finished or when you're done working, you can exit the virtual environment by running:
```shell
(env) $ deactivate
```

By following the steps above, you should be able to successfully install and configure RegArch. If you encounter any issues during the installation process, please contact us for further assistance.
