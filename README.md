# lulwimi
Lulwimi is a Siswati document analyser.  This was developed during a hackathon organized by DHASA and Digimethods on 16-18 October 2024.  As such, the code in its initial version is not the most beautiful.

The aim of the project was to see if (or in how far) it is possible to develop a tool that can analyse textual documents written in Siswati.  A few language independent analyses are implemented (some token and sentence counting, word clouds, and LDA including visualization).  Language specific analyses were limited by the limited availability of tools for Siswati.  (A few tools are available, but only work on Windows.)

To improve the quality of the word clouds, a stopword list for Siswati was developed based on the novel "Tinyembeti", written by Jabulani Pato.  (We also looked at using exam texts from the data collection that can be found on the SADiLaR repository: https://hdl.handle.net/20.500.12185/568 (however, in the end this was not used).  Improvements on the stopword list are probably possible.

The code only serves as an initial attempt to perform some textual analysis.  Due to the limited time available and the limited avaialbility of tools, this can most likely be improved massively.  Also note that essentially no work has been done to create a visually pleasing output.
