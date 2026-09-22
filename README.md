# task_modul_4

Acesta este un repozitoriu pentru taskul 4.

In acest task am urmat pasii pe care i-am vazut in modulul patru.

1.In primul script python, data_cleaning, am facut urmatoarele:

a). Am incarcat bibliotecile python necesare si setul de date.

b). Am curatat unde era nevoie numele coloanelor de spatii.

c). Am afisat tipurile de date, valorile lipsa, duplicatele din csv.

d). Am curatat coloana Product title, convertind la string, scapand de spatii, scapand de randuri lipsa si sau goale.

e). Am curatat coloana Category Label in acelasi fel ca si anterior.

f). Am standardizat in coloana Category label unde era nevoie, de ex("fridge":"Fridges").

g). Am scapat de duplicate.

h). Am afisat un sumar al setului de date dupa curatare.

i). Am afisat intr-un plot distributia categoriilor.

j). Am facut Feature Engineering asupra coloanei Product title in care am stabilit cum sa calculam numarul de litere din titlu, numarul de cuvinte, numarul de numere din titlu, numarul de caractere speciale si cel mai lung cuvant.

k). Am pregatit datele pentru machine learning prin setarea X si y.

l). Am separat datele de invatare de cele de testare.

m). Am definit modele de invatare automata ce urmeaza a fi testate: Logistic Regression, Naive Baynes, Linear Svm, Random forest.

n). Am antrenat modelele.

o) Am comparat si vizualizat intr-un plot performanta modelelor.

p).Am selectat un model, am afisat matricea sa de confuzie, apoi am facut un rezumat.


2.In al doilea script python, train_model, am facut urmatoarele:

a). Am incarcat bibliotecile python necesare si fisierul csv.

b). Am curatat setul de date precum am facut in primul script.

c). Stabilind modelul de invatare,Linear svm, l-am aplicat doar pe acesta pe setul de date.

d). Am salvat modelul de invatare intr-un fisier pkl pentru a-l folosi in al treilea script.


3. In al treilea script, predict_category, am facut urmatoarele:

a) Am incarcat bibliotecile python necesare si fisierul pkl.

b) Am creat o bucla infinite while prin care cerem input utilizatorului pentru un produs.

c) Am creat conditie de iesire si am scapat de spatiile inutile din eventualul titlu introdus de utilizator.

d) Am creat conditie pentru input gol.

e) Transformam input utilizatorilor in Series pentru a putea aplica modelul de predictie si afisam rezultatul.


4. Odata ce am facut aceste script-uri, am facut un repozitoriu git in care am uploadat urmatoare:
 
a). In folderul data am uploadat dateset csv si modelul salvat pkl.

b). Separat in google colab am transpus cele 3 scripturi python in notebook-uri diferite care sa faca acelasi lucru si le-am salvat din colab in Github, in folderul Notebooks.

c) Procedura prin care sa creez un fisier pkl prin Colab mi s-a parut complicata asa ca am uploadat pkl local obtinut pur si simplu.


