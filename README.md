$$
i \hbar \partial_t \Psi = -\frac{\hbar^2}{2m} \nabla^2  \Psi \\ ~ \\ 

\partial_t \Psi = \frac{i \hbar}{2m} \sum_{d = 0}^D (\partial_{x^d}^2) \Psi  \\ ~ \\ 

\frac{\Psi_{next} - \Psi}{\delta_t} = \frac{i \hbar}{2m} \sum_{d = 0}^D \frac{\Psi_{+x^d} + \Psi_{-x^d} - 2 \Psi }{\delta_x^2}\\ ~ \\ 

\Psi_{next} = \Psi + \left( \frac{i \hbar \delta_t}{2m \delta_x^2}\sum_{d}^D ( \Psi_{+x^d} + \Psi_{-x^d}) \right) - \left( \frac{i \hbar \delta_t D}{m \delta_x^2} \right) \Psi \\ ~ \\

\Psi_{next} = \left( \frac{i \hbar \delta_t}{2m \delta_x^2}\sum_{d}^D ( \Psi_{+x^d} + \Psi_{-x^d}) \right) + \left( 1 - \frac{i \hbar \delta_t D}{m \delta_x^2} \right) \Psi \\ ~ \\
$$

The stability condition is then:
$$
1 - D\frac{\delta_t}{\delta_x^2} \frac{\hbar}{m} \gt 0
$$

Using $D=2$, this leads to:
$$
\frac{m}{2\hbar} \gt \frac{\delta_t}{\delta_x^2}
$$

Choosing units so that $m = \hbar = 1$:

$$
\frac{\delta_x^2}{2} \gt \delta_t \Leftrightarrow \delta_x \gt \sqrt{2 \delta_t}
$$