# COVID
<H1>Quantifying the effect of quarantine control in Covid-19 infectious spread using machine learning</H1>
<H2> Introduction</H2>
 <p align="justify">The incumbent COVID 19 infection has emerged as a major pandemic affecting millions of people and curtailing economies across 
 the world. Many countries are enforcing strict quarantine measures by locking down the whole country in order to mitigate 
 the spread of corona virus COVID 19. On the positive note, these lockdown measures were effective in slowing down the 
 infection rate but had other serious consequences like unemployment. More than 20 million people have filed for unemployment 
 benefits in USA alone and many companies including small and medium enterprises are at the risk of bankruptcy. 
 With no clear directives on the extent of lockdown due to unavailability of accurate model to simulate the infection spread, 
 it appears that the stay at home orders will continue forever.</p>

<p align="justify"> In this project, we attempt to forecast the number of days the lockdown measure has to be implemented 
to curtail the infection spread. We used the COVID-19 infection and recovery data from different countries to build a 
data driven model by leveraging artificial neural network with a compartmental based mathematical model, 
SIR (Susceptible, Infected and Recovered), a widely used model in epidemiology to study the spread of infectious diseases.</p>

<H2>Problem formulation</H2>
We explore two models SIR and SIRT to model the COVID 19 spread. However, we study SIR model in depth which is a simplified 
version of SIRT to understand the dynamics of the problem with respect to optimization.
<H3>SIR model</H3>
<p align="justify"> SIR(S - Susceptible,I - Infected ,R - Recovered) is a compartmental model in epidemiology that simplifies 
the mathematical modelling of infectious disease spread. The SIR model is governed by the following ODE's:</p>
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Barray%7D%7Brcl%7D%20%5Cfrac%7B%5Cpartial%20S%28t%29%29%7D%7B%5Cpartial%20t%7D%20%26%3D%26%20-%5Cfrac%7B%5Cbeta%20S%28t%29I%28t%29%7D%7BN%7D%5C%5C%20%5Cfrac%7B%5Cpartial%20I%28t%29%7D%7B%5Cpartial%20t%7D%20%26%3D%26%20%5Cfrac%7B%5Cbeta%20S%28t%29I%28t%29%7D%7BN%7D-%5Cgamma%20I%28t%29%5C%5C%20%5Cfrac%7B%5Cpartial%20R%28t%29%7D%7B%5Cpartial%20t%7D%20%26%3D%26%20%5Cgamma%20I%28t%29%20%5Cend%7Barray%7D">
 </p>
