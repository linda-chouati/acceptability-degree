# Acceptability Degree Visualization

This project implements the **weighted h-categorizer semantics** and provides an interactive visualization of the **acceptability degree spaces** of argumentation frameworks.

It was developed as part of the continuous assessment for the Argumentation and Decision course (2025).

---

## Overview

The application computes the **acceptability degrees** of arguments in a weighted framework using the _h-categorizer semantics_.  
Then, it samples random weight vectors, transforms them into acceptability vectors, and builds the **convex hull** that represents the boundary of all possible acceptability values.

An interactive visualization is provided through **Streamlit**, allowing users to explore 1D, 2D, and 3D acceptability spaces with sliders and adjustable parameters.

---

## Project Structure

- src :
  - framework.py : definition of arguments, attacks, and parsing
  - hc_semantics.py : implementation of the weighted h-categorizer
  - sampler.py : random weight generation and transformation
  - hull.py : convex hull computation using SciPy
    visuals.py : interactive visualization (2D / 3D)
- app.py : main streamlit application
- requirements.txt : list of dependencies

---

## Installation

Make sure you have **Python 3.9+** installed.

1. Clone or download the repository.
2. Install the required dependencies: pip install -r requirements.txt
3. Run the Streamlit application: : streamlit run app.py

---

## Online Application

The online version is available here: https://acceptability-degreegit-ruxr6h7b4tnywziht9uujn.streamlit.app
