# Projektbericht: Fussgaengerstreifen-Erkennung in Luftbildern

Status: 2026-06-14

## 1. Ziel des Projekts

Ziel des Projekts ist es, in einem `25 m x 25 m` grossen Luftbildausschnitt zu
erkennen, ob ein Fussgaengerstreifen vorhanden ist. Die Bilder stammen aus
Swisstopo/SWISSIMAGE. Die geforderte Aufgabe ist damit eine binaere
Bildklassifikation: `crosswalk` oder `no_crosswalk`.

Das Projekt loest diese Klassifikation mit einem Segmentierungsansatz. Das
eigene Modell `CrossMaskNet` sagt zuerst eine Maske voraus, die den vermuteten
Fussgaengerstreifen im Bild markiert. Aus der Groesse dieser Maske wird danach
die finale Klassifikation abgeleitet. Wenn die vorhergesagte Maskenflaeche ueber
dem Schwellwert liegt, wird das Bild als `crosswalk` klassifiziert, sonst als
`no_crosswalk`.

Dieser Ansatz erfuellt die Klassifikationsaufgabe und macht die Entscheidung
gleichzeitig besser nachvollziehbar, weil sichtbar wird, auf welchen Bildbereich
sich die Vorhersage stuetzt.

## 2. Eckdaten des Datensatzes

Die Grundlage ist der Datensatz `sam3-500k-masks-v1`. Er enthaelt Metadaten und
Labels fuer `500,000` ausgewaehlte Luftbild-Kacheln aus 14 Schweizer Staedten.
Die grossen Rohbilder werden nicht im Git-Repository gespeichert, sondern bei
Bedarf reproduzierbar von Swisstopo heruntergeladen.

| Kennzahl | Wert |
|---|---:|
| Ausgewaehlte Kacheln | `500,000` |
| Source-positive Kacheln | `8,815` |
| Source-negative Kacheln | `491,185` |
| Metadaten-Shards | `489` |
| Quell-Szenen | `489` |
| Abgedeckte Staedte | `14` |

Die Labels wurden mit SAM3 als Pseudo-Labeling-Werkzeug erzeugt. SAM3 wurde nur
fuer die Erstellung der Trainingsdaten verwendet. Das finale Modell ist nicht
SAM3, sondern das in diesem Projekt entwickelte und trainierte CrossMaskNet.

Fuer das Training wurde aus dem grossen Metadatensatz ein kleinerer
CrossMaskNet-Export mit `5,800` Beispielen erstellt:

| Split | Crosswalk | No Crosswalk | Total |
|---|---:|---:|---:|
| Train | `1,600` | `3,406` | `5,006` |
| Validation | `189` | `193` | `382` |
| Test | `211` | `201` | `412` |
| Total | `2,000` | `3,800` | `5,800` |

Die Datenqualitaet ist fuer das Training geeignet, aber die automatische
Label-Erzeugung fuehrt zu typischen schwierigen Faellen. Dazu gehoeren
Dachstrukturen, Parkplatzmarkierungen, Schatten, Vegetation und andere
Strassenmarkierungen, die Fussgaengerstreifen aehneln koennen. Diese Faelle
wurden gezielt als schwierige Negativbeispiele genutzt, um das Modell robuster
zu machen.

## 3. Architektur des Modells

Das finale Modell heisst `CrossMaskNet v4`. Es ist ein eigenes, kompaktes
U-Net-aehnliches Convolutional Neural Network in PyTorch und wurde ohne
vortrainierte Gewichte trainiert.

Die Eingabe besteht aus vier Kanaelen:

- roter Bildkanal;
- gruener Bildkanal;
- blauer Bildkanal;
- Road-Context-Kanal.

Der Road-Context-Kanal gibt dem Modell Zusatzinformation darueber, wo
strassenaehnliche Bereiche liegen. Das ist hilfreich, weil Fussgaengerstreifen
typischerweise auf oder nahe bei Strassen vorkommen und nicht auf Daechern,
Wiesen oder Baeumen.

CrossMaskNet besitzt eine Encoder-Decoder-Struktur. Der Encoder verdichtet das
Bild und lernt zunehmend abstrakte Merkmale. Der Decoder baut daraus wieder eine
Maske in Bildaufloesung auf. Skip Connections uebertragen feine Details vom
Encoder in den Decoder, damit kleine Strukturen wie Streifen nicht verloren
gehen. Am Ende erzeugt eine `1 x 1`-Faltung eine ein-kanalige
Wahrscheinlichkeitsmaske fuer Fussgaengerstreifen.

Die eigentliche Klassifikation wird aus dieser Maske abgeleitet. Die
Maskenflaeche wird mit einem Schwellwert verglichen. Der Standardwert im Projekt
ist `0.005`. Niedrigere Werte fuehren zu mehr positiven Vorhersagen, hoehere
Werte zu strengeren positiven Entscheidungen.

## 4. Optimierung

Die wichtigste Verbesserung kam nicht nur aus der Architektur, sondern aus der
Datenarbeit. Fruehe Modellversionen erkannten viele echte Fussgaengerstreifen,
erzeugten aber auch falsche Treffer bei aehnlichen Mustern. Deshalb wurden
schwierige Negativbeispiele gezielt in das Training aufgenommen. Dazu zaehlen
vor allem Dachmuster, Parkplatzlinien, Schatten und andere Markierungen.

Weitere Optimierungen waren:

| Optimierung | Wirkung |
|---|---|
| Hard Negatives | Reduzieren falsche positive Vorhersagen bei visuell aehnlichen Mustern. |
| Road-Context-Kanal | Hilft dem Modell, plausible Strassenbereiche zu erkennen. |
| Confidence- und Maskenfilter | Entfernen schwache Pseudo-Labels aus dem Trainings-Export. |
| Dice-artiger Segmentierungsverlust | Hilft bei kleinen Zielobjekten, weil Fussgaengerstreifen nur wenige Pixel belegen. |
| Threshold-Kalibrierung | Steuert den Kompromiss zwischen Precision und Recall. |

Trainiert wurde CrossMaskNet v4 fuer 12 Epochen mit Bildgroesse `128`, Batch
Size `64`, vier Eingabekanaelen und einer Learning Rate von `0.0007`. Als
Optimierer wurde AdamW verwendet. Das beste Modell wird anhand der
Validierungsmetriken gespeichert und anschliessend auf dem Testsplit bewertet.

## 5. Resultate

CrossMaskNet v4 erreicht auf dem Testsplit folgende Werte:

| Metrik | Wert | Bedeutung |
|---|---:|---|
| Accuracy | `0.961165` | Anteil korrekter `crosswalk` / `no_crosswalk` Entscheidungen. |
| Precision | `0.957746` | Wie oft eine positive Vorhersage wirklich positiv ist. |
| Recall | `0.966825` | Wie viele positive Beispiele gefunden werden. |
| Positive Dice | `0.794408` | Ueberlappung der vorhergesagten Maske mit der Zielmaske. |
| Positive IoU | `0.658937` | Strengere Ueberlappungsmetrik fuer die Maskenqualitaet. |

Die Resultate zeigen, dass das Modell die Klassifikationsaufgabe zuverlaessig
loest und gleichzeitig brauchbare Masken erzeugt. Fuer die praktische
Interpretation ist besonders wichtig, dass Precision und Recall ueber den
Schwellwert beeinflusst werden koennen. Ein strengerer Schwellwert reduziert
falsche positive Treffer, kann aber schwach sichtbare Fussgaengerstreifen
verpassen. Ein niedrigerer Schwellwert findet mehr positive Beispiele, erzeugt
aber eher zusaetzliche positive Vorhersagen.

## 6. Persoenliche Erkenntnisse

Die wichtigste Erkenntnis ist, dass die Datenqualitaet fuer dieses Projekt
mindestens so wichtig war wie die Modellarchitektur. Schon die erste
CrossMaskNet-Version lernte sinnvolle Muster, hatte aber Probleme mit
schwierigen Negativbeispielen. Durch gezieltes Hinzufuegen solcher Faelle wurde
das Modell deutlich robuster.

Eine zweite Erkenntnis ist, dass Segmentierung die Klassifikation besser
erklaerbar macht. Ein reines Klassifikationsmodell liefert nur eine Ja/Nein-
Antwort. CrossMaskNet zeigt zusaetzlich eine Maske und damit den Bereich, auf
dem die Entscheidung basiert.

Die dritte Erkenntnis betrifft automatisches Labeling. SAM3 war sehr hilfreich,
um schnell einen grossen Datensatz aufzubauen. Trotzdem muss man automatisch
erzeugte Labels filtern und mit schwierigen Beispielen arbeiten, damit das
trainierte Modell nicht nur einfache Faelle loest.

## 7. Fazit

Das Projekt erfuellt die Aufgabenstellung, weil es Luftbild-Kacheln in
`crosswalk` und `no_crosswalk` klassifiziert. Der eingereichte Ansatz geht
darueber hinaus, indem er zuerst eine Maske vorhersagt und daraus die
Klassifikation ableitet. Dadurch bleibt die Loesung kompatibel mit der
geforderten Klassifikation und ist gleichzeitig besser interpretierbar.

Der Datensatz wurde mit SAM3-Pseudo-Labels aufgebaut. Das eingereichte Modell
ist CrossMaskNet v4, ein eigenes PyTorch-Modell, das in diesem Repository
trainiert und ausgewertet wird.

