Requires Python 3 and some dependencies. If you use
[pipenv](https://github.com/kennethreitz/pipenv) you can install them using

    $ pipenv install

Assignment 2:

 * hospital1.py contains the first 4 questions and can be run as a script
 * rand.py contains the others, but you need to import it as a module and call
   the correct function

Assignment 3:

 * hospital2.py is an updated version of the simulation
 * statplot.py will produce stats and plots
 * hospital2.py can be run like::
    ```
    pipenv run python hospital2.py tasks1n2
    pipenv run python hospital2.py task3
    pipenv run python hospital2.py task4
    ```

  * statplot.py can be run like::
    ```
    pipenv run python statplot.py task3.dat task3.png
    ```
