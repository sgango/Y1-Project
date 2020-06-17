[![CodeFactor](https://www.codefactor.io/repository/github/sgango/y1-project/badge?s=3379d2c96618c35ee74140bce62f972780f0a727)](https://www.codefactor.io/repository/github/sgango/y1-project)

# Imperial Physics: Year 1 Project

***Daniel Buguks and Srayan Gangopadhyay***  
A Python module for visualising the trajectories of charged particles in electric and magnetic fields.  

## Basic usage

**These are temporary instructions. Ease of use will be improved soon.**

* Download `y1project.py` (currently in `/ntbks`) and `environment.yml`.
* Set up a Python environment with the packages listed in `environment.yml` (see [conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)).
* Open a Jupyter notebook in the same directory as `y1project.py`.
* Paste this code into a cell:

```python
import y1project as pj

# PARAMETERS
r0 = [1e7,1e7, 0]  # initial position
v0 = [1e7, 1e7, 0]  # initial velocity
q = pj.e  # charge
m = pj.m_pr  # mass
h = 0.01  # step size
end = 20  # t-value to stop integration
size = [1e10, 1e10, 5e6]  # simulation dimensions

def B_field(r):
    """Returns the components of the magnetic
    field, given a 3D position vector."""
    return [1e-9, 1e-16, 2e-8]

def E_field(r):
    """Returns the components of the electric
    field, given a 3D position vector."""
    return [0,0,0]

r, v = pj.rk4(pj.lorentz, r0, v0, h, end, E_field, B_field, q, m, size)
pj.plot_or_anim(r, int((end/h)/100), *pj.plotsetup(r), anim=True)
pj.printout(r, v)  # print table of stats
# pj.textout(r, v)  # export data to csv
```

* Run, edit, repeat!

### Expected output

Something like this:

<img src="https://user-images.githubusercontent.com/25332542/84928613-9bac6080-b0c6-11ea-83af-1e79ade8e365.png" width="400">

### Known issues

* Seeing a weird "Hbox" text output instead of the progress bars?  
    * Follow the instructions in [this GitHub issue](https://github.com/tqdm/tqdm/issues/394#issuecomment-384743637).
    
Any other problems? Let us know by [opening an issue](https://github.com/sgango/Y1-Project/issues/new).

---

## For developers: repository management

**/dev**: for unfinished code, or anything that won't be used in the final product  
**/ntbks**: Jupyter Notebooks for testing and demonstration  
**/y1project**: finished modules go in here

- **Create a new branch for whatever you're working on and merge it with a pull request.**
