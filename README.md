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
<p align = "center">
<img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Barray%7D%7Brcl%7D%20%5Cfrac%7B%5Cpartial%20S%28t%29%29%7D%7B%5Cpartial%20t%7D%20%26%3D%26%20-%5Cfrac%7B%5Cbeta%20S%28t%29I%28t%29%7D%7BN%7D%5C%5C%20%5Cfrac%7B%5Cpartial%20I%28t%29%7D%7B%5Cpartial%20t%7D%20%26%3D%26%20%5Cfrac%7B%5Cbeta%20S%28t%29I%28t%29%7D%7BN%7D-%5Cgamma%20I%28t%29%5C%5C%20%5Cfrac%7B%5Cpartial%20R%28t%29%7D%7B%5Cpartial%20t%7D%20%26%3D%26%20%5Cgamma%20I%28t%29%20%5Cend%7Barray%7D">
</p>
<p align="justify">where β is the infection rate, γ is the recovery rate and N = S(t) + I(t) + R(t). The infection rate β and recovery rate γ are obtained by minimizing the loss function provided below:
 <p align="middle"><img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Barray%7D%7Brcl%7D%20L%28%5Cbeta%2C%5Cgamma%29%3D%20%7C%7Clog%28I_%7Bpred%7D%29-log%28I_%7Bactual%7D%29%7C%7C%5E2&plus;%7C%7Clog%28R_%7Bpred%7D%29-log%28R_%7Bactual%7D%29%7C%7C%5E2&plus;%20%5Clambda_%7B1%7D%7C%7Clog%28I_%7Bpred%7D%29%7C%7C%5E%7B2%7D&plus;%5Clambda_%7B2%7D%7C%7Clog%28R_%7Bpred%7D%29%7C%7C%5E%7B2%7D%20%5Cend%7Barray%7D"></p>
 
 <p align="justify">where I<sub>{pred}</sub> and R<sub>pred</sub> are the infected and recovered cases predicted by the model and I<sub>actual</sub> and R<sub>actual</sub> are the actual infected and recovered cases obtained from John Hopkins corona virus tracker. In summary, the SIR optimization problem can be defined as follows</b>
 
 <b>Objective</b> : Minimize loss function<br>
 <b> Design variables</b> : β, the infection rate and γ, the recovery rate<br>
 <b> Constraints</b> : β ∈ (0,1) and γ ∈ (0,1)

<H3>SIRT model</H3> 
<p align="justify">
 In addition to the above model, new terms  quarantined population (T(t)) and quarantine strength Q(t) are used to quantify the quarantine control.These terms coupled with SIR model are governed by the following set of ordinary differential equations. The quarantine strength Q(t) is highly dependent on local quarantine policies, when modelling the COVID-19 spread for a country. So, a neural network is used to approximate the quarantine strength. The governing equations for SIRT is given by:</p>
 <p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Barray%7D%7Brcl%7D%20%5Cfrac%7B%5Cpartial%20S%28t%29%7D%7B%5Cpartial%20t%7D%20%26%3D%26%20-%5Cfrac%7B%5Cbeta%20S%28t%29I%28t%29%7D%7BN%7D%5C%5C%20%5Cfrac%7B%5Cpartial%20I%28t%29%7D%7B%5Cpartial%20t%7D%20%26%3D%26%20%5Cfrac%7B%5Cbeta%20S%28t%29I%28t%29%7D%7BN%7D-%28%5Cgamma%20&plus;Q%28t%29%29I%28t%29%5C%5C%20%5Cfrac%7B%5Cpartial%20R%28t%29%7D%7B%5Cpartial%20t%7D%20%26%3D%26%20%5Cgamma%20I%28t%29%5C%5C%20%5Cfrac%7B%5Cpartial%20T%28t%29%7D%7B%5Cpartial%20t%7D%20%26%3D%26%20Q%28t%29I%28t%29%5C%5C%20Q%28t%29%20%26%3D%26%20NN%28W%2Ct%29%20%5Cend%7Barray%7D">
  </p>
  where NN is the neural network, W represents the weights of the neural network and t is the time.<br>
  <p>The neural network weights W, infection rate β and recovery rate γ are obtained by minimizing the loss function provided below:</p>
  <p align="center">
 <img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Barray%7D%7Brcl%7D%20L%28W%2C%5Cbeta%2C%5Cgamma%29%3D%20%7C%7Clog%28I_%7Bpred%7D%29-log%28I_%7Bactual%7D%29%7C%7C%5E2&plus;%7C%7Clog%28R_%7Bpred%7D%29-log%28R_%7Bactual%7D%29%7C%7C%5E2&plus;%5C%5C&plus;%5Clambda_%7B1%7D%7C%7Clog%28I_%7Bpred%7D%29%7C%7C%5E%7B2%7D&plus;%5Clambda_%7B2%7D%7C%7Clog%28R_%7Bpred%7D%29%7C%7C%5E%7B2%7D%20%5Cend%7Barray%7D">
 </p>
 <b>Objective</b> : Minimize loss function<br>
 <b> Design variables</b> : β, the infection rate and γ, the recovery rate and W (neural net parameters)<br>
 <b> Constraints</b> : β ∈ (0,1) and γ ∈ (0,1)<br>
 
 <H2>Data modelling</H2>
 <p align="justify">
 In this report, we plan to study the COVID-19 spread across 4 countries USA, UK, Germany and Italy. The data we collected range from January 22, 2020 to May 5, 2020. The initial data point corresponds to the outbreak in a country. We define the infection spread  as outbreak when a country's infected cases rise to more than 500. The data is collected from John Hopkin's Coronavirus Resource Center.</p>
 <H3> Coupling ODEs with Optimizer</H3>
 <H4> SIR model </H4>
 <p align="justify">The ODE equations are solved by using the solver odeint from <i>scipy</i> package and it is integrated with optimizer as follows:</p>
 <ol>
  <li>For the initial β and γ, the ODE's are solved by the odeint solver from <i>scipy</i> package.</li>
  <li>The solver returns the predicted infected and recovered cases to loss function.</li>
  <li>The optimizer computes gradients using finite difference method by solving the ODEs again for β+∂β and γ+∂γ</li>
 <li>The design variables are updated using the gradients without violating the constraints and this process is repeated until the minima for the loss function is found.</li>
</ol> 
 <H4> SIRT model </H4>
 <p align = "justify">Similar to SIR model, the ODE equations for SIRT are solved by using the solver odeint from <i>scipy</i> package and it is integrated with optimizer and neural network as follows:</p>
 
 <ol>
  <li>For initial time t, Q(t) is computed using initial neural network weights, W.</li>
  <li>With the computed Q(t), initial β and γ, the ODE's are solved by the odeint solver from <i>scipy</i> package.</li>
  <li>The solver returns the predicted infected and recovered cases to loss function.</li>
 <li>The  optimizer computes gradients using finite difference method by solving the ODE's again for β+∂β and γ+∂γ and W+∂W (w1+∂w1,w2+∂w2...). It is important to note that changing weights W, lead to changes in Q(t).</li>
 <li>The design variables are updated using the gradients without violating the constraints and the process is repeated until the minima for the loss function is found.</li>
</ol> 

<H2> Proposed Optimization Algorithm </H2>
 <H3> SIR model </H3>
<p align="justify">The SIR function seems to be continuous, as the results generated by solving ODE's are continuous. So, we decided to use a gradient based optimizer SLSQP, which handles constraints well.</p>
<H3> SIRT model </H3>
<p align ="justify">Coupling neural network to the existing SIR model might led to discontinuity, so we used a gradient free optimizer, Nelder-Mead to reduce the mean squared loss function.</p>

<H2> References </H2>
The study is an extension of the work carried out by Dandekar <i>et al.</i> in,<br>
<p align="justify"><i><b>Dandekar, R. and Barbastathis, G., 2020. Quantifying the effect of quarantine control in Covid-19 infectious spread using machine learning. medRxiv.</b></i></p>


 
 
  
  

