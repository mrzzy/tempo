# Tempo
_by tempura for Splash Awards 2019_

## Introduction 
Tempo is JS library aimed at enhancing security on web by providing transparent
2 factor authentication through keystroke detection when typing.

This allows developers to enhance security, without introducing the hassle that 
typically comes with enhanced security. This means Tempo is simple and ready to
deploy for developers. Additionality, this enanced security comes at no cost to
users, which will encounter no change.

> **We want to enhance web security, without introducing hassle.**

## Design
We propose a non-intrusive and transparent bot detection/2-factor
authentication  solution. While the user types their credentials, keystroke
dynamic features, such as typing speed, are analysed.


### Models
For 2-factor authentication, an autoencoder
model (similar to Facenet[1]) will be used to produce a embedding for keystroke
dynamic features. The embedding is compared to a reference to determine if the
keystrokes are characteristic of the user.

### Stack
![Tempo Stack](./assets/stack.png)
TODO: refine & elaborate

### Project Structure
- src - project source code
    - keyprint - user fingerprinting through keystroke dynamics
    - dataprep - utilities used for preparing data
- data
    - datasets - datasets
- lib - JS library to intergrate tempo with your site

## Datasets
#### Keyboard
- `cmu_bench/` [Keystroke Dynamics - Benchmark Data Set CMU](https://www.cs.cmu.edu/~keystroke/)
- `cmu_eval/` [Free vs. Transcribed Text for Keystroke-Dynamics Evaluations](http://www.cs.cmu.edu/~keystroke/laser-2012/)
- `beihang_keys/` [The BeiHang Keystroke Dynamics Database](http://mpl.buaa.edu.cn/detail1.htm)
- `keystroke100/` [Keystoke100 Dataset](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_keystroke100.html)
- `greyc_keys/` [GREYC-KeyStroke Dataset](https://web.archive.org/web/20181017094316/http://www.epaymentbiometrics.ensicaen.fr/greyc-keystroke-dataset)
- `greyc_web/` [GREYC-Web Based KeyStroke Dynamics Dataset](http://www.labri.fr/perso/rgiot/ressources/GREYC-WebDataset.html)

#### Touch Screen
- `mobikey/` [ The MOBIKEY Keystroke Dynamics Password Database](https://ms.sapientia.ro/~manyi/mobikey.html#rawdata
- `android_keys/` [Keystroke Dynamics - Android platform](https://ms.sapientia.ro/~manyi/keystroke.html)
