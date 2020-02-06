# Aim:

You are required to implement Simulated Annealing (SA), Genetic Algorithm (GA) and Tabu Search (TS) algorithm to solve 48 capitals of the US (ATT48) Travelling Salesman Problem (TSP).

The original Simulated Annealing paper published in Science can be found here (https://pdfs.semanticscholar.org/beb2/1ee4a3721484b5d2c7ad04e6babd8d67af1d.pdf). A tutorial on Tabu search is available here (https://www.ida.liu.se/~zebpe83/heuristic/papers/TS_tutorial.pdf)
 
# Requirements:

1. You can use any programming languages to complete this assignment. However, if you want to use languages other than Matlab/Octave, you should make your program executable/runnable. For example, if you use Java, you need to compile it. If you use Python, make sure it can be run in a python online IDE such as TutorialPoint (https://www.tutorialspoint.com/online_python_ide.php).
2. Your program should be able to read in the ```att48.tsp``` file.
3. Calculate distance based on pseudo-Euclidean distance. Please read this document (```tsp95.pdf```) (Section 2.5) to learn how to calculate the pseudo-Euclidean distance. In order to check the correctness of your implementation of the distance calculation, you can download the file (```att48.opt.tsp```) which gives you the optimal tour with the optimal distance of 10628.  
4. Implement the Simulated Annealing (SA) Genetic Algorithm (GA) and Tabu Search (TS) algorithms. For TS, you can implement the simplest version with the simplest Tabu list which only remembers recently visited solutions.
5. Execute maximum 30 trial runs for each algorithm to tune the parameters (Hint: You probably need to do some literature search to find the appropriate parameter ranges). Record how you tune the parameters.
6. After obtaining good parameters, execute 30 independent runs with 3000 iterations for each algorithm. Record the average distance and standard deviation from the results over the 30 runs for each algorithm, respectively.
7. Compare the results for these two algorithms statistically using a Wilcoxon signed-rank test (https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test). If you do not know the statistical hypothesis test, please read this article (https://en.wikipedia.org/wiki/Statistical_hypothesis_testing).
8. Write a report to report your results. The report should include:

  * Brief introduction of the SA, GA and TS algorithms. You need to justify your design decisions, e.g., encoding scheme for GA, and explain these algorithms by using a flowchart and pseudo-code.
  * Discuss what the parameters are and how you tuned them.
  * You should also list all the average result and standard deviations obtained from the 30 runs of the algorithms
  * Discuss how you compare the results obtained by SA and Tabu search statistically.

 
# Marking Scheme (total 25 points):

* Correct import the ATT48.tsp file and calculation of the distance. (2 marks).
* Correct implementation of SA (5 marks)
* Correct implementation of GA (5 marks)
* Correct implementation of the TS algorithm (5 marks)
* Good results, e.g., +/-10% from the optimal result (3 marks)
* Report: Satisfied requirement 8 (5 marks).

 
# Submission guideline:

You must submit your source code, compiled binary executable (if applicable), and a PDF report. 
