# RadioLab 0.15 — récepteur SDR en Python pour Airspy R2 + SpyVerter

## Nouveau dans la 0.15 : quatre décodeurs

Tous dans la section « Décodage numérique », en USB. **Vérifiés sur signaux de synthèse seulement** : aucun n'a
encore été essayé sur une vraie émission.

- **WSPR** (balises radioamateurs de faible puissance). Régler la fréquence « cadran » de la bande : 474,2 kHz ;
  1836,6 ; 3568,6 ; 7038,6 ; 10138,7 ; 14095,6 ; 18104,6 ; 21094,6 ; 28124,6 kHz. Les balises occupent alors
  1400 à 1600 Hz audio. Le décodage a lieu une fois toutes les deux minutes, vers la seconde 114 des minutes
  paires : **l'horloge du PC doit être à l'heure** (à une ou deux secondes près), et il faut avoir écouté la
  période entière. Chaque ligne donne l'heure UTC, la fréquence audio, le décalage en secondes, l'indicatif, le
  locator et la puissance annoncée. Limites : messages ordinaires seulement (indicatif simple + locator à
  4 caractères), pas de compensation de la dérive en fréquence, jamais comparé à WSJT-X sur de vrais signaux. Ne fonctionne pas
  sur un fichier rejoué (l'heure ne correspond plus).
- **Hellschreiber** (14063 kHz, 7030-7040 kHz, 3580 kHz) : s'accorder pour que la note tombe sur la « Fréquence
  audio » choisie, ou cliquer sur le signal. Le texte apparaît sur une bande, sous la cascade, tracé en double :
  c'est normal, et c'est à l'œil de le lire.
- **DCF77** (77,5 kHz) : en USB, cliquer sur la porteuse (RadioLab s'accorde 1 kHz plus bas). Les bits s'affichent
  un par seconde ; au bout d'une minute complète, la date et l'heure apparaissent. Filtre étroit conseillé.
- **Allouis 162 kHz** (l'ancien émetteur de France Inter, qui ne diffuse plus que l'heure) : même manœuvre. Ici
  c'est la phase de la porteuse qui porte les bits. Décodeur expérimental : l'émetteur ajoute d'autres données
  entre les tops, que mes signaux d'essai ne reproduisent pas.

Pour ces deux signaux horaires il faut compter une à deux minutes avant la première heure affichée.

## Nouveau dans la 0.14

**Confort**
- **Filtre à la souris** : sur le spectre, tirer un bord de la zone claire change la largeur du filtre ; tirer le
  milieu déplace la fréquence.
- **Raccourcis clavier** : ← → un pas (Ctrl : dix pas), Page haut / bas change de mode, Espace coupe le son,
  M mémorise, R enregistre le son, L verrouille, P met un fichier en pause, F1 affiche la liste.
- **S-mètre** en barre (S0 à S9+60 dB) dans la barre du haut. Il se règle dans « Affichage » (Étalonnage S-mètre) :
  sans étalonnage il ne sert qu'à comparer des signaux entre eux, car le niveau dépend du gain.
- **Noms des mémoires** affichés sur le spectre, **maintien des crêtes** (courbe orange) et **moyennage**.
- **Étalonnage automatique du SV PPM** (section « Source ») : s'accorder près d'une porteuse de fréquence connue,
  cliquer, indiquer sa fréquence exacte en hertz ; la correction est calculée et appliquée.

**Lecteur de fichiers I/Q** : sous la cascade, bouton pause et curseur de position.

**Charge de calcul** dans la barre d'état. Au-delà de 90 %, augmenter la décimation ou baisser le débit.

**Nouveaux décodeurs** (section « Décodage numérique », en USB) — vérifiés sur signaux de synthèse seulement :
- **NAVTEX** : 518 kHz (anglais) ou 490 kHz (langue locale). S'accorder 1 kHz sous la fréquence, ou cliquer sur le
  signal. Le sens des deux tonalités est trouvé automatiquement.
- **Fax météo HF** (120 lignes/min, IOC 576), par exemple le service météo allemand sur 3855, 7880 et 13882,5 kHz.
  Cliquer **au milieu** du signal (RadioLab s'accorde 1,9 kHz plus bas). L'image se trace sous la cascade, même prise
  en cours de route ; le curseur « Cadrage » la recale dans ce cas. Une image complète est enregistrée d'office en PNG.
- **SSTV** : Martin M1/M2, Scottie S1/S2/DX, PD90/120/180, Robot 36, reconnus par leur en-tête. Fréquences
  habituelles : 14230 kHz, 7171 kHz, 3730 kHz ; la station spatiale émet parfois en PD120 sur 145,800 MHz : passer alors
  en NFM, le décodeur SSTV fonctionne aussi dans ce mode.
- **Constellation PSK31/63** à côté du tracé du niveau : deux taches nettes = signal propre.

**Pour les curieux** : `fabriquer_exe.bat` construit un dossier `dist\RadioLab` avec `RadioLab.exe`, utilisable
sans installer Python (non essayé de mon côté : je n'ai pas de Windows). Le code de l'interface est maintenant
réparti entre `gui.py` (fenêtre) et `widgets.py` (thème et petits éléments).


## Nouveau dans la 0.13 : interface réorganisée

- **Thème sombre** et spectre rempli sous la courbe.
- **Barre du haut** : ☰ (masquer / afficher les réglages), marche-arrêt, fichier I/Q, coupure du son, volume,
  puis un **grand afficheur de fréquence chiffre par chiffre** : molette sur un chiffre, ou clic sur sa moitié
  haute (+) ou basse (−). Le champ de saisie directe en hertz reste à côté.
- **Panneau de gauche** : tous les réglages, auparavant en onglets à droite, sont regroupés en **sections
  repliables** (Source, Radio, Affichage, Anti-bruit, Mémoires, Scanner, Décodage…). Un clic sur l'en-tête
  replie ou déplie ; l'état est mémorisé d'une session à l'autre.
- Les **boutons de mode** sont en tête de la section « Radio ».
- Les curseurs Zoom / Contraste / Plage / Décalage restent à droite du spectre.

Logiciel de bureau dans l'esprit de SDR# : spectre, cascade, accord à la souris,
démodulation AM / LSB / USB / CW / NFM / WFM, anti-bruit (NB, NR, notch
automatique), enregistrement audio et I/Q, lecture de fichiers I/Q, mémoires.

## Installation (Windows)

1. Installer **Python 3.10 ou plus récent, en 64 bits** (python.org), en cochant
   « Add python.exe to PATH ».
2. Dans le dossier de RadioLab, ouvrir une invite de commandes :
   `pip install -r requirements.txt`
3. Fournir la bibliothèque du récepteur : **`airspy.dll` en 64 bits** et les DLL
   qui l'accompagnent (`libusb-1.0.dll`, DLL pthread). Les copier dans le dossier
   de RadioLab, ou laisser RadioLab demander l'emplacement au premier démarrage
   (il le mémorise). Elles font partie des outils hôtes Airspy
   (`airspyone_host`) ; une copie se trouve aussi en général dans le dossier de
   SDR#. Une DLL 32 bits ne peut pas être chargée par un Python 64 bits.
4. Fermer SDR# (un seul logiciel peut utiliser l'Airspy à la fois), puis
   double-cliquer sur `RadioLab.bat` ou lancer `python -m radiolab`.

## Utilisation

- **▶ Airspy** démarre la réception. **SpyVerter** coché : la HF est reçue via le
  convertisseur (+120 MHz ajoutés automatiquement ; on tape la
  vraie fréquence, par exemple 7 100 000 Hz). Décoché : réception directe de 24
  à 1800 MHz.
- **Correction SV PPM** : compense l'erreur de l'oscillateur 120 MHz du SpyVerter
  (1 ppm = 120 Hz). Se règle sur une porteuse de fréquence connue ; active
  seulement quand SpyVerter est coché.
- **Bias-tee** : case séparée. Cochée, l'Airspy envoie une tension continue sur
  la prise d'antenne pour alimenter le SpyVerter (ou un préamplificateur). À
  décocher si le SpyVerter a sa propre alimentation ou si rien n'est à alimenter.
- **Débit et décimation** (onglet « Réception ») : « Débit » choisit 2,5 ou
  10 MS/s ; « Décimation » (2 à 32) ne garde que la bande centrale, à un débit
  divisé d'autant. À 10 MS/s décimé par 16, la bande affichée fait 500 kHz et le
  spectre est 16 fois plus détaillé à taille de FFT égale. Les enregistrements
  I/Q sont faits après décimation, donc bien plus légers. La FM large demande
  au moins 200 kS/s après décimation.
- **Fenêtres « Spectre FI » et « Spectre audio »** : deux cases à cocher dans
  « Affichage et accord ». La première montre le voisinage immédiat de la
  fréquence écoutée (axe en écart par rapport à elle, zone claire = filtre) ; la
  seconde montre le son après démodulation et traitements.
- **Curseurs à droite du spectre** : Zoom (resserre la vue autour de la
  fréquence écoutée), Contraste (cascade automatique), Plage et Décalage
  (hauteur et haut de l'échelle du spectre, en dB).
- **Niveaux de la cascade** (« Affichage et accord ») : automatiques (calés sur
  le bruit de fond) ou manuels, avec « Cascade min » (niveau affiché en bleu
  nuit) et « Cascade max » (niveau affiché en rouge).
- **Modes DSB et RAW** : double bande latérale ; I/Q brut sur les deux voies
  audio (I à gauche, Q à droite) pour un logiciel externe.
- **AM synchrone** (« Écoute ») : démodule l'AM avec la porteuse reconstituée ;
  le son reste propre quand la porteuse s'évanouit en ondes courtes.
- **Note CW** : hauteur de la note en mode CW, de 300 à 1200 Hz.
- **Décalage (autre convertisseur)**, **Inverser le spectre**, **Supprimer la
  raie centrale** : groupe « Récepteur ».
- **Caler la fréquence sur la grille du pas** : onglet « Affichage » ; décoché,
  la molette et les flèches ajoutent simplement le pas.
- **Groupes de mémoires** : liste déroulante au-dessus des mémoires ; le scanner
  ne parcourt que le groupe affiché.
- **Filtre asymétrique, type et ordre du filtre** (« Écoute ») : « Bord bas » et
  « Bord haut » règlent chaque côté du filtre séparément ; le type (fenêtre) et
  l'ordre (nombre de coefficients, 0 = automatique) règlent la raideur des flancs.
- **Anti-fading** (avec l'AM synchrone) : garde la bande latérale la plus forte.
- **Binaural** : en BLU et CW, les deux oreilles reçoivent le signal décalé d'un
  quart de période ; au casque, les signaux se séparent dans l'espace.
- **Second récepteur (VFO B)** (« Outils ») : une seconde fréquence dans la bande
  reçue, avec son mode et son volume, mélangée ou sur l'oreille droite. Trait
  bleu clair sur le spectre.
- **TimeShift** (« Outils ») : garde les dernières secondes en mémoire (jusqu'à
  512 Mo : environ 25 s à 2,5 MS/s, 100 s à 625 kS/s) et permet d'écouter et de
  voir le signal avec un retard choisi. La mémoire est vidée quand le récepteur
  change de fréquence centrale.
- **Packet AX.25** (« Décodage ») : APRS / packet VHF à 1200 bauds (mode NFM, par
  exemple 144,800 MHz) et packet HF à 300 bauds (USB ou LSB, tonalités autour de
  la « Fréquence audio », 1700 Hz conseillé). Seules les trames au CRC juste
  sont affichées.
- **Mode TETRA (analyse seulement)** : bouton « TETRA ». RadioLab démodule un
  canal TETRA de 25 kHz (pi/4-DQPSK, 18 000 symboles/s), mesure la qualité et
  l'écart de fréquence, compte les salves reconnues à leurs séquences
  d'apprentissage, et lit la balise de synchronisation : pays, réseau, code
  couleur, numéros de trame et de multitrame. Le reste (appels, voix) n'est PAS
  décodé ; aucun son. Vérifié sur un enregistrement réel de deux porteuses.
- **Molette sur le spectre ou la cascade** : la fréquence avance ou recule du pas
  choisi. **Ctrl + molette** : zoom.
- **Zoom et finesse d'analyse** : Ctrl + molette sur le spectre pour zoomer,
  glisser pour se déplacer. Le spectre et la cascade sont recalculés pour la seule bande
  affichée, avec une FFT d'autant plus grande que le zoom est fort (jusqu'à un
  million de points, soit 2,4 Hz par point à 2,5 MS/s). « Finesse (FFT) » permet
  d'imposer une taille ; « Fenêtre » choisit la pondération (Blackman-Harris,
  Hann, Hamming, Blackman, Kaiser, Flat-top, Poisson, rectangulaire : une
  info-bulle décrit chacune). Une FFT très fine observe le signal plus longtemps :
  l'affichage se rafraîchit moins vite et réagit plus lentement.
- **Accord** : clic sur le spectre ou la cascade, glisser le trait rouge, taper
  la fréquence, ou molette / flèches avec le pas choisi. Le pas se choisit de
  1 Hz à 1 MHz (dont 6,25 kHz et ses multiples, 8,33 kHz, 9 kHz) ; le clic et les
  flèches calent la fréquence sur la grille de ce pas.
- **Gain** : 0 à 21. Monter jusqu'à ce que le bruit de fond se soulève de
  quelques dB, pas plus.
- **Anti-bruit** : NB contre les claquements, NR contre le souffle (la méthode
  MMSE-LSA est la plus agréable), ANF contre les sifflements de porteuse (inactif
  en CW, où il supprimerait le signal).
- **FM de radiodiffusion** (mode WFM) : stéréo automatique quand la station émet
  le pilote, nom de la station et texte RDS dans l'onglet « Analyse ».
- **Plan de bandes** : zones colorées sur le spectre (vert : amateur, orange :
  radiodiffusion, bleu : aviation, violet : divers) et nom de la bande à côté des
  modes. Indicatif, limites à vérifier selon le pays.
- **Scanner** (onglet « Mémoires ») : parcourt les mémoires ou une plage de
  fréquences avec le pas choisi. Il s'arrête quand le niveau du canal dépasse le
  **squelch**, reste 2 s après la fin du signal, puis repart. Régler d'abord le
  squelch juste au-dessus du « Niveau du canal » lu sans signal.
- **CTCSS** : en FM étroite, la tonalité subaudible est affichée dans « Analyse ».
- **Verrouiller la fréquence** : bloque l'accord contre les clics involontaires.
- **Spectre audio** : onglet « Analyse ».
- **DCS** : le code (par exemple D023N) s'affiche à côté du CTCSS. Un même signal
  correspond à un code « normal » et à un autre « inversé » : RadioLab affiche le
  premier trouvé.
- **Modes numériques HF** (onglet « Décodage ») : RTTY 45,45 et 50 bauds, PSK31,
  PSK63 et morse (CW), décodés en texte. Passer en USB (ou LSB), choisir le mode,
  puis **cliquer sur le signal dans la cascade** : RadioLab s'accorde pour que ce
  signal tombe sur la « Fréquence audio » du décodeur (1000 Hz par défaut, trait
  violet du spectre audio). En mode CW, il suffit de cliquer sur le signal.
  Le PSK se recale seul à quelques hertz près ; en RTTY, cocher « Inverser » si
  le texte est illisible. Les premiers caractères d'une émission peuvent manquer.
  Dès qu'un décodeur est choisi, un bandeau apparaît au-dessus du spectre : tracé
  du niveau du signal (avec le seuil de décision en vert pour le CW) et texte
  décodé sur toute la largeur. En CW, la vitesse et le seuil sont automatiques ;
  on peut les imposer dans l'onglet « Décodage » (le seuil se lit sur le tracé).
- **WSJT-X, fldigi (FT8, FT4 et autres modes)** : régler RadioLab en USB, envoyer
  le son vers un câble audio virtuel (« Outils > Sortie audio »), cocher
  « Commande réseau », puis choisir dans le logiciel le poste « Hamlib NET
  rigctl » à l'adresse 127.0.0.1:4532. Il règle alors la fréquence de RadioLab.
- **POCSAG** (onglet « Décodage ») : en mode NFM, messages de radiomessagerie à
  512, 1200 et 2400 bit/s, avec l'adresse du destinataire.
- **Filtres audio** (onglet « Outils ») : graves, aigus, et notchs manuels (on
  donne la fréquence audio à supprimer ; le spectre audio aide à la repérer).
- **Sortie audio** : choix du périphérique. Avec un câble audio virtuel et
  « Sortie brute du discriminateur », RadioLab alimente un décodeur externe
  (DSD+ et autres) comme le font les greffons passerelles de SDR#.
- **Commande réseau** : serveur au format rigctl sur 127.0.0.1:4532. Dans
  Gpredict, déclarer une radio sur cette adresse pour la correction Doppler.
  Seules les commandes de base sont gérées (fréquence, mode).
- **Mémoires** : import et export CSV (nom ; fréquence en Hz ; mode ; groupe).
- **Journal d'activité** : chaque arrêt du scanner ajoute une ligne (heure,
  fréquence, niveau, tonalité) au fichier CSV du jour.
- **Enregistrements programmés** : « Programmer… » dans « Outils » ; RadioLab
  doit être en réception à l'heure dite.
- **Enregistrement** : fichiers WAV dans le dossier personnel. Un enregistrement
  I/Q à 2,5 MS/s pèse 10 Mo par seconde (limite du format : environ 6 minutes).
- **Sans récepteur** : « Ouvrir un fichier I/Q… » relit un enregistrement, ou
  `python -m radiolab.cli fichier_IQ.wav --freq 7100000 --mode LSB --nr -o audio.wav`.

## Organisation du code

| Fichier | Rôle |
| --- | --- |
| `radiolab/dsp/filters.py` | filtres, décimation polyphase, rééchantillonnage |
| `radiolab/dsp/receiver.py` | chaîne complète I/Q → audio, démodulateurs, AGC |
| `radiolab/dsp/noise.py` | noise blanker, filtre adaptatif NLMS, NR spectrale |
| `radiolab/dsp/fmradio.py` | FM stéréo, décodeur RDS, détection CTCSS |
| `radiolab/dsp/spectrum.py` | spectre pour l'affichage |
| `radiolab/dsp/digital.py` | horloge de bits, DCS, POCSAG |
| `radiolab/dsp/hfdigi.py` | RTTY, PSK31/63, morse |
| `radiolab/dsp/packet.py` | packet AX.25 (APRS 1200 bauds, HF 300 bauds) |
| `radiolab/dsp/tetra.py` | TETRA : démodulation et repérage des salves |
| `radiolab/dsp/audiofx.py` | graves, aigus, notchs manuels |
| `radiolab/scanner.py`, `radiolab/bandplan.py` | scanner, plan de bandes |
| `radiolab/netcontrol.py`, `radiolab/planner.py` | commande réseau ; CSV, journal, programmation |
| `radiolab/sources/` | Airspy (ctypes sur libairspy), fichier WAV I/Q |
| `radiolab/radio.py` | liaison source → récepteur → carte son, enregistrements |
| `radiolab/gui.py` | interface (PySide6 + pyqtgraph) |
| `tests/` | banc d'essai : `python -m pytest` (demande `pytest`, et `scipy` pour un essai) |

Le traitement du signal n'utilise que NumPy et se lit de haut en bas dans
`receiver.py` ; chaque bloc est commenté.

## Ce qui a été vérifié, et ce qui ne l'a pas été

Vérifié par le banc d'essai (signaux synthétiques) :
- sélection de bande latérale, réjection des signaux voisins (> 55 dB), note CW,
  AM, NFM, WFM ; sortie identique quelle que soit la taille des blocs ;
- noise blanker, notch automatique (> 30 dB sur une porteuse), NR spectrale
  (gain de SNR d'environ 7 à 8 dB sur un signal « type voix » dans du bruit blanc) ;
- FM stéréo (séparation des voies > 30 dB), RDS (nom, texte, code PI décodés
  jusqu'à 22 dB de rapport signal/bruit), CTCSS, scanner : sur un émetteur
  synthétique fabriqué d'après les normes, pas sur de vraies stations ;
- DCS et POCSAG : les codes correcteurs retrouvent les mots de référence publiés
  (DCS 023 et 346, mots de synchronisation et de repos POCSAG) ; le décodage
  complet est essayé sur des émissions synthétiques, pas sur de vrais signaux ;
- RTTY, PSK31/63 et CW : émissions synthétiques avec bruit, à travers toute la
  chaîne de réception. La table Varicode du PSK31 est vérifiée contre les
  propriétés publiées ; la table Baudot (chiffres et ponctuation) vient de mémoire ;
- commande réseau essayée avec un client de test, pas avec Gpredict, WSJT-X ni
  fldigi : la description du poste envoyée à Hamlib est reconstituée de mémoire ;
- lecture et écriture des fichiers, enchaînement des réglages de l'interface,
  appels à la bibliothèque Airspy (contre une fausse bibliothèque).

**Non vérifié** : le développement s'est fait sans Airspy, sans écran et sans
carte son. La fenêtre n'a jamais été affichée, la vraie `airspy.dll` n'a jamais
été appelée, aucun son n'a été écouté. Le premier lancement peut donc révéler
des défauts : copier le message d'erreur complet permet de les corriger vite.

## Ce que SDR# fait et que cette version ne fait pas

WSPR, Hellschreiber, SSTV, OmniRig et CAT par port série, Meteor, TV analogique,
FT8 / FT4 natifs (passer par WSJT-X), décodeurs numériques intégrés (TETRA, APCO, DSD : passer par la sortie
brute et un décodeur externe), réduction de bruit neuronale, compensation fine de l'écart d'horloge entre le
récepteur et la carte son (ici : on jette un peu d'audio quand la file déborde).
