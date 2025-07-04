# One-Shot-Learning-for-Cancer-Detection-through-Blood-Analysis
<p><strong>Autrice:</strong> Rosaria Leone</p>

<h2>Abstract</h2>
<p>
Il cancro rappresenta una delle maggiori sfide globali per la salute, con un impatto significativo su mortalità e qualità della vita.
La diagnosi precoce aumenta le probabilità di trattamento efficace, ma le attuali tecniche presentano limiti di invasività, costi elevati e scarsa efficacia nelle fasi iniziali.
In questo scenario, l’analisi di biomarcatori proteici ematici si afferma come soluzione non invasiva per identificare firme molecolari di malignità.
</p>
<p>
Questa ricerca esplora l’applicazione delle reti neurali siamesi all’analisi dei biomarcatori ematici, offrendo un approccio innovativo per distinguere sottili pattern nelle firme proteiche tra individui sani e malati. Le reti siamesi, grazie alla loro architettura avanzata, permettono l’individuazione precoce di tumori multipli, utilizzando biomarcatori accessibili da semplici prelievi di sangue e offrendo scalabilità verso nuovi tipi di cancro senza completo riaddestramento del modello.
</p>
<p>
L'approccio proposto supera i limiti delle biopsie liquide tradizionali, unendo sensibilità, specificità ed anche la localizzazione precisa del tessuto tumorale. I risultati ottenuti puntano a rendere lo screening oncologico più efficace, sostenibile e accessibile a livello di popolazione.
</p>

<h2>Struttura del Progetto</h2>
<ul>
  <li><strong>main.py</strong>: Implementazione principale della rete neurale siamese per classificazione multi-classe su dati oncologici.</li>
  <li><strong>One_vs_n-1.py</strong>: Rimozione iterativa di ogni classe dal dataset di Train e di Test.</li>
  <li><strong>fine_tuning.py</strong>: Tuning e addestramento del modello branch con ottimizzazione degli iperparametri.</li>
  <li><strong>test_oneshot.py</strong>: Rimozione iterativa di ogni classe dal dataset di Train.</li>
  <li><strong>utility.py</strong>: Funzioni di supporto per costruzione modelli, metriche (accuratezza/sensibilità/specificità), visualizzazione e test.</li>
  <li><strong>Acc_Sen_Spe_AUC.py</strong>: Calcolo delle metriche di performance (sensibilità, specificità, AUC) specifico per il contesto di classificazione binaria.</li>
  <li><strong>preprocessing.py</strong>: Preprocessing dati: lettura, pulizia, encoding, imputazione e scaling di dataset Excel.</li>
  <li><strong>testgpu.py</strong>: Verifica disponibilità GPU per TensorFlow.</li>
  <li><strong>Data.xlsx</strong>: Dataset biomarcatori e dati clinici pazienti.</li>
  <li><strong>requirements.txt</strong>: Librerie e versioni richieste.</li>
</ul>

<h2>Requisiti</h2>
<ul>
  <li>Python &ge; 3.8</li>
  <li>TensorFlow &ge; 2.6</li>
  <li>Keras</li>
  <li>scikit-learn</li>
  <li>pandas</li>
  <li>imbalanced-learn (SMOTE)</li>
  <li>matplotlib</li>
  <li>Consulta <code>requirements.txt</code> per tutte le dipendenze.</li>
</ul>

<h2>Installazione</h2>
<ol>
  <li>Clona il repository:<br>
    <code>git clone https://github.com/your-username/your-repo.git</code><br>
    <code>cd your-repo</code>
  </li>
  <li>Crea un ambiente virtuale e attivalo (consigliato):<br>
    <code>python -m venv venv</code><br>
    <code>source venv/bin/activate</code> (Linux/Mac)<br>
    <code>venv\Scripts\activate</code> (Windows)
  </li>
  <li>Installa le dipendenze:<br>
    <code>pip install -r requirements.txt</code>
  </li>
</ol>

<h2>Esempio di utilizzo</h2>
<ul>
  <li>Per l’addestramento principale:<br>
    <code>python main.py</code>
  </li>
  <li>Per il preprocessing dati:<br>
    <code>python preprocessing.py</code>
  </li>
  <li>Per test specifici:<br>
    <code>python test_oneshot.py</code><br>
    <code>python One_vs_n-1.py</code>
  </li>
  <li>Per verificare la GPU:<br>
    <code>python testgpu.py</code>
  </li>
</ul>

<h2>Dataset</h2>
<p>
  <strong>Data.xlsx</strong>: contiene biomarcatori ed informazioni cliniche di pazienti, utilizzato per addestramento, validazione e test.
</p>

<h2>Output</h2>
<ul>
  <li>Risultati e metriche vengono salvati in file <code>.json</code> e <code>.txt</code> a seconda degli script eseguiti.</li>
  <li>Grafici di performance e confusion matrix generati in automatico.</li>
</ul>


<h2>Contatti e Contributi</h2>
<p>Per contributi, segnalazioni o suggerimenti, aprire una issue o inviare una pull request.<br>
Autrice: Rosaria Leone – <i>rosarialeone500@gmail.com/https://www.linkedin.com/in/rosaria-leone/</i></p>
