# Home Return Predictor

Integrazione custom per Home Assistant che **impara quando ogni persona di casa torna**
e pubblica, per persona e per la casa intera, la **probabilità di rientro** entro un
orizzonte (default 90 minuti), l'**orario stimato di arrivo (ETA)** e quanto il
modello è **affidabile** (maturità). Pensata per pre‑riscaldare le zone con
[Versatile Thermostat](https://github.com/jmcollin78/versatile_thermostat) (VTherm)
tramite il blueprint incluso, ma i sensori sono utilizzabili in qualsiasi automazione.

- Dominio: `home_return_predictor`
- Nessuna dipendenza esterna, nessun servizio cloud: tutto gira in locale.
- Si installa tramite HACS (repository custom, privato) o copia manuale.

---

## Indice

1. [Cosa fa / cosa non fa](#cosa-fa--cosa-non-fa)
2. [Come funziona](#come-funziona)
3. [Probabilità e maturità](#probabilità-e-maturità)
4. [Modalità: learning, live, auto](#modalità-learning-live-auto)
5. [Installazione](#installazione)
6. [Configurazione e opzioni](#configurazione-e-opzioni)
7. [Entità](#entità)
8. [Evento e servizi](#evento-e-servizi)
9. [Blueprint](#blueprint)
10. [Gerarchia di controllo con VTherm](#gerarchia-di-controllo-con-vtherm)
11. [Priorità delle decisioni](#priorità-delle-decisioni)
12. [Esempi multi‑zona e notte](#esempi-multi-zona-e-notte)
13. [Plugin termici: confronto e combinazione consigliata](#plugin-termici-confronto-e-combinazione-consigliata)
14. [Esempi di automazioni](#esempi-di-automazioni)
15. [I primi giorni di apprendimento](#i-primi-giorni-di-apprendimento)
16. [Modello sfidante scikit-learn (opzionale)](#modello-sfidante-scikit-learn-opzionale)
17. [Per sviluppatori: simulazione e test](#per-sviluppatori-simulazione-e-test)
18. [Limiti noti](#limiti-noti)

---

## Cosa fa / cosa non fa

**Fa**

- Ricostruisce dallo storico del recorder (fino a 120 giorni) le assenze di ogni
  `person` e continua a imparare in tempo reale.
- Stima, mentre una persona è fuori, la probabilità che rientri entro l'orizzonte e
  l'ETA, combinando le **abitudini** con la **posizione GPS** (se disponibile).
- Misura da sola quanto è affidabile (Brier score contro una baseline) e passa da
  *learning* a *live* quando lo merita.
- Aggrega a livello casa: probabilità che qualcuno rientri, prima ETA, "qualcuno sta
  tornando".
- Fornisce un blueprint che sceglie il **preset** di ogni zona VTherm.

**Non fa**

- **L'integrazione non comanda nessuna entità.** Pubblica solo sensori ed eventi.
  Solo il blueprint (o le tue automazioni) agiscono sui termostati, e sei tu a
  scegliere per ogni zona quali preset usare.
- Non imposta mai temperature: il blueprint cambia solo i preset di VTherm.
- Non modella la termica delle stanze (quanto ci mette una stanza a scaldarsi): per
  questo vedi [Plugin termici](#plugin-termici-confronto-e-combinazione-consigliata).
- Non sostituisce il rilevamento finestre o le sicurezze di VTherm.

---

## Come funziona

Per ogni persona il modello combina due fonti di informazione.

1. **Prior dalle abitudini.** Le assenze passate (< 10 min scartate come rumore GPS,
   > 36 h trattate come viaggi) vengono pesate per somiglianza con quella in corso:
   - ora di uscita simile (kernel circolare, ±75 min circa);
   - stesso giorno della settimana / stessa classe feriale‑festivo (calendario
     festività opzionale);
   - recenza (emivita configurabile, default 28 giorni), così un cambio di
     abitudini viene assorbito in poche settimane.

   La probabilità è una stima di sopravvivenza: "tra le assenze simili che erano
   ancora in corso dopo lo stesso tempo, quante sono finite entro l'orizzonte?".
   Funziona anche con i **turni**: chi esce alle 6 torna a un'ora diversa da chi esce
   alle 14, e il modello lo distingue dall'ora di uscita. Con pochi dati simili la
   stima viene attenuata verso un valore medio, invece di diventare estrema.
2. **Evidenza GPS.** Dalle posizioni recenti (precisione peggiore di 250 m scartata)
   calcola la velocità di avvicinamento a casa e la classifica (si allontana, fermo,
   si avvicina piano/veloce…). Ogni classe ha un **likelihood ratio auto‑calibrato**:
   impara da solo quanto "si sta avvicinando" significhi davvero "sta tornando" per
   quella persona (chi passa spesso vicino casa senza rientrare abbassa il peso di
   quel segnale).
3. **Fusione.** `logit(P) = logit(prior) + log(LR)`, limitata tra 1% e 99%. Nessuna
   regola scritta a mano per tipo di persona: abitudini nette dominano, abitudini
   confuse lasciano più spazio al GPS.
4. **ETA.** Media pesata (per incertezza) tra l'ETA da abitudine e l'ETA da
   distanza/velocità quando ci si muove a più di 5 km/h.

Una persona **in casa** ha probabilità 100% e `returning` spento.

---

## Probabilità e maturità

Sono due numeri diversi e vanno letti insieme.

| | Domanda a cui risponde | Dove |
|---|---|---|
| **Probabilità** | "Quanto è probabile che torni entro 90 minuti, adesso?" | `sensor.<p>_return_probability` |
| **Maturità** | "Quanto mi posso fidare di queste probabilità, in generale?" | `sensor.<p>_model_maturity` |

La maturità cresce con:

- **quantità di dati**: numero di assenze (pesate per recenza);
- **bravura dimostrata**: il modello registra le proprie previsioni e, a posteriori,
  le confronta con la realtà. Calcola il **Brier score** (errore quadratico medio,
  più basso = meglio) e lo confronta con una **baseline** volutamente semplice: la
  probabilità di rientro calcolata solo da "da quanto tempo è fuori", senza ora di
  uscita né giorno. Il *Brier skill score* (`1 − brier / baseline`) misura quindi
  quanto valgono le abitudini imparate; servono almeno 30 previsioni verificate.

Una probabilità del 90% con maturità 20% è un'ipotesi; con maturità 85% è una
previsione su cui agire.

---

## Modalità: learning, live, auto

Ogni persona ha un `select.<p>_mode`:

| Modalità | Comportamento |
|---|---|
| `learning` | Tutto viene calcolato e pubblicato (anche `returning`), ma con attributo `advisory: true`. Pensato per osservare senza agire. |
| `live` | Previsioni pienamente operative, `advisory: false`. |
| `auto` (default) | Passa a live quando la maturità supera la soglia (default 70%) e torna a learning se scende sotto soglia − 10 punti (**isteresi**: niente avanti e indietro). |

I sensori casa hanno `mode: live` se almeno una delle persone rilevanti (quelle che
stanno tornando, altrimenti tutte) è in live, e `advisory` di conseguenza.

Il blueprint VTherm ha l'input **"Agisci sulle previsioni solo in modalità live"**,
**attivo di default**. Finché il modello impara, il blueprint si comporta come una
classica automazione presenza + calendario + letto (quindi può sostituire da subito
le vecchie automazioni della zona), ma **ignora il rientro previsto**: niente
pre‑riscaldamenti su previsioni immature. Quando la persona passa in live, il
pre‑riscaldo si attiva da solo. Disattiva l'input se vuoi il pre‑riscaldo da subito,
accettando qualche accensione inutile nelle prime settimane.

Se preferisci tenere le tue automazioni attuali durante l'apprendimento, non creare
ancora le automazioni del blueprint: osserva `binary_sensor.<p>_returning` e la
maturità, e attiva il blueprint quando sei convinto.

---

## Installazione

### Tramite HACS (repository privato)

Il repository `giogandola98/ha-home-return-predictor` è **privato**. HACS può
scaricarlo solo se il token GitHub configurato in HACS ha accesso al repository:

1. HACS usa il token GitHub ottenuto durante la sua configurazione iniziale
   (login GitHub). L'**account GitHub di quel token deve avere accesso** al repo
   privato (proprietario o collaboratore). Se HACS è stato configurato con un altro
   account, riconfiguralo (Impostazioni → Dispositivi e servizi → HACS → Riconfigura)
   oppure aggiungi quell'account come collaboratore del repo.
2. HACS → menu ⋮ → **Repository personalizzati**.
3. URL: `https://github.com/giogandola98/ha-home-return-predictor`, categoria
   **Integrazione** → Aggiungi.
4. Cerca "Home Return Predictor" → Scarica → **riavvia Home Assistant**.

### Manuale

Copia `custom_components/home_return_predictor/` in
`<config>/custom_components/home_return_predictor/` e riavvia.

Requisiti: Home Assistant 2025.1 o successivo, `recorder` attivo, persone (`person`)
con tracker (meglio se l'app Companion con GPS).

---

## Configurazione e opzioni

Impostazioni → Dispositivi e servizi → Aggiungi integrazione → **Home Return
Predictor**: scegli le persone e la zona di casa (default `zone.home`). Lo storico
viene importato subito dal recorder.

Opzioni (Configura):

| Opzione | Default | Significato |
|---|---|---|
| Soglia di rientro | 0.6 | Probabilità sopra la quale `returning` si accende |
| Soglia di maturità | 0.7 | Maturità per il passaggio automatico a live |
| Orizzonte | 90 min | Finestra della domanda "torna entro…?" |
| Emivita | 28 giorni | Quanto velocemente si dimenticano le vecchie abitudini |
| Intervallo di aggiornamento | 2 min | Ricalcolo periodico (oltre agli aggiornamenti sugli eventi) |
| Calendario festività | — | `calendar.*` opzionale: i giorni con eventi sono trattati come festivi |
| Modello sfidante scikit-learn | off | Installa scikit-learn e addestra in background un modello alternativo, usato solo se migliore (vedi [sotto](#modello-sfidante-scikit-learn-opzionale)) |

---

## Entità

Per ogni persona `<p>`:

| Entità | Tipo | Contenuto |
|---|---|---|
| `binary_sensor.<p>_returning` | binary | On se la probabilità supera la soglia. Attributi: `probability`, `eta`, `mode`, `advisory`, `reason`, `active_model` |
| `sensor.<p>_return_probability` | % | Probabilità di rientro entro l'orizzonte (100 se in casa) |
| `sensor.<p>_return_eta` | timestamp | ETA stimata (pubblicata mentre la persona è fuori) |
| `sensor.<p>_model_maturity` | % | Maturità. Attributi: `episodes`, `brier_score`, `baseline_brier_score`, `brier_skill_score`, `active_model`, `candidate_brier`, `challenger_enabled`, `sklearn_available` |
| `sensor.<p>_return_reason` | testo | Spiegazione leggibile della stima corrente |
| `select.<p>_mode` | select | `auto` / `learning` / `live` |

Casa:

| Entità | Contenuto |
|---|---|
| `sensor.house_return_probability` | % che almeno una persona fuori rientri entro l'orizzonte; **100 se qualcuno è in casa** |
| `sensor.house_next_return_eta` | ETA più vicina tra le persone fuori con probabilità ≥ 25% (attributo `eta_person`), così il blueprint può usare anche la fascia "probabilità media" |
| `binary_sensor.house_someone_returning` | On se almeno una persona sta tornando |

Tutte le entità casa hanno gli attributi `occupied`, `persons_home`,
`returning_persons`, `mode`, `advisory` (il binary sensor anche `returning_names`).

---

## Evento e servizi

**Evento** `home_return_predictor_return_detected`, emesso quando una persona passa a
"sta tornando":

```yaml
person: person.giorgio
name: Giorgio
probability: 0.82
eta: "2026-10-12T18:40:00+02:00"
mode: live
advisory: false
```

**Servizi**

| Servizio | Effetto |
|---|---|
| `home_return_predictor.retrain` | Ricostruisce il modello dallo storico (persone opzionali) |
| `home_return_predictor.reset` | Cancella il modello appreso per le persone indicate |
| `home_return_predictor.export_diagnostics` | Esporta lo stato del modello (coordinate oscurate) |

---

## Blueprint

| File | Scopo |
|---|---|
| `blueprints/automation/home_return_predictor/vtherm_zone_preheat.yaml` | Sceglie il preset VTherm di **una zona** (un'automazione per zona) |
| `blueprints/automation/home_return_predictor/presence_bridge.yaml` | Copia "qualcuno sta tornando" (e opzionalmente "qualcuno è in casa") in un `input_boolean`, per strumenti che accettano solo un'entità on/off |

**Importazione.** Il pulsante/URL "Importa blueprint" di Home Assistant **non funziona
con un repository privato** (HA scarica senza autenticazione). Copia i file YAML in:

```
<config>/blueprints/automation/home_return_predictor/
```

poi Impostazioni → Automazioni → Blueprint → (ricarica) → crea un'automazione per
ogni zona.

### Input del blueprint VTherm

| Input | Default | Note |
|---|---|---|
| Termostati VTherm | — | Una o più entità `climate` della zona |
| Tipo di zona | `day` | `day` (giorno), `night` (notte), `always` (sempre usata, es. bagno) |
| Calendario zona | — | Helper `schedule`: quando la zona è usata. Vuoto = sempre |
| Sensore occupazione zona | — | `binary_sensor`: se on, con qualcuno in casa forza comfort |
| Interruttore letto/notte | — | `input_boolean`/`switch`/`binary_sensor` |
| Probabilità casa | `sensor.house_return_probability` | |
| ETA casa | `sensor.house_next_return_eta` | |
| Sensore "qualcuno torna" | `binary_sensor.house_someone_returning` | Attributi `mode`/`advisory` |
| Fonte "qualcuno in casa" | `zone.home` | Stato zona = numero di persone; anche person/group/binary_sensor |
| Agisci solo in live | **on** | Vedi [Modalità](#modalità-learning-live-auto) |
| Minuti di pre‑riscaldo fissi | 30 | |
| Sensore minuti di pre‑riscaldo | — | Se numerico vince sul valore fisso (es. `sensor.<zona>_predicted_duration` di ha‑preheat) |
| **Preset comfort** | `comfort` | Zona in uso, letto (zone notte), rientro molto probabile |
| **Preset probabilità media** | `eco` | Rientro con probabilità media |
| **Preset assenza** | `frost` | Nessuno in casa e nessun rientro probabile |
| **Preset zona non in uso** | `eco` | Qualcuno in casa ma zona fuori calendario |
| **Preset notte (zone giorno)** | `eco` | Zone giorno con interruttore letto acceso |
| Soglia alta | 70% | ≥ soglia → preset comfort |
| Soglia media | 40% | ≥ soglia → preset probabilità media |
| Isteresi | 5% | Evita il ping‑pong tra preset |
| Scrivi decisioni nel logbook | off | Utile per capire perché ha scelto un preset |

Tutti i preset sono scelti **per zona** tra `frost`/`eco`/`comfort`/`boost`, oppure
scrivendo un altro nome di preset VTherm. Se il preset scelto non esiste su un
termostato, quel termostato non viene toccato.

> **Perché `frost` e non `away`?** In VTherm il preset `away` è nascosto e **non
> selezionabile** quando la funzione presenza di VTherm è disattivata (come deve
> essere, vedi sotto). Per questo l'assenza usa `frost` di default.

Il blueprint gira ogni 2 minuti, all'avvio di HA e a ogni cambio delle entità in
ingresso. La logica Jinja è identica a
`custom_components/home_return_predictor/hrp_core/preset_logic.py`, coperta da test.

---

## Gerarchia di controllo con VTherm

Regola fondamentale: **un solo "cervello" cambia i preset di una zona.**

```
Home Return Predictor  →  chi torna, quando, con che certezza (solo sensori)
          │
Blueprint VTherm zona  →  UNICO che chiama climate.set_preset_mode
          │
Versatile Thermostat   →  temperature dei preset, regolazione, finestre, sicurezze
```

- **Disattiva la funzione presenza di VTherm** (menu *Funzioni* del termostato:
  `use_presence_feature` off) e anche il **movimento**. Se restano attive, VTherm
  cambia temperature in base alla presenza per conto suo e le due logiche si
  contraddicono (e `away` diventerebbe l'unico modo di gestire l'assenza).
- **Lascia attivi** rilevamento finestre e sicurezza (safety) di VTherm, ed
  eventuale gestione potenza: il blueprint li rispetta e non tocca il termostato
  mentre sono attivi.
- Il blueprint **non imposta mai temperature**. Cambiare a mano la temperatura in
  VTherm porta il preset a `none`: al giro successivo il blueprint rimetterà il
  preset calcolato. Se vuoi un controllo manuale prolungato, disattiva
  l'automazione della zona.
- Disattiva le vecchie automazioni che cambiano i preset delle stesse zone quando
  attivi il blueprint: due automazioni sulla stessa zona si contraddicono.

---

## Priorità delle decisioni

Per ogni termostato della zona, dalla più alta alla più bassa:

| # | Condizione | Risultato |
|---|---|---|
| 1 | Protezioni VTherm: finestra aperta (senza bypass), safety, sovraccarico potenza, preset `safety`/`power`, HVAC spento, termostato bloccato | **Non tocca nulla** |
| 2 | Interruttore letto acceso **e** qualcuno in casa | Zone giorno → preset notte; zone notte/sempre → comfort |
| 3 | Qualcuno in casa | Zona in uso (occupazione on, nessun calendario, o calendario attivo ora) → comfort; altrimenti → preset zona non in uso |
| 4 | Rientro previsto (se consentito dalla modalità) | Se adesso ≥ ETA − minuti di pre‑riscaldo (o l'ETA è già passata) **e** il calendario della zona è attivo all'ETA: p ≥ alta → comfort, p ≥ media → preset media, altrimenti → assenza |

Regole aggiuntive:

- Con qualcuno in casa, il rientro di un'altra persona può solo **alzare** il preset,
  mai abbassarlo (ordine: assenza/frost < eco < comfort < boost).
- Con nessuno in casa vale il punto 4, o il preset assenza.
- **Isteresi**: se il preset attuale è comfort, resta comfort finché p ≥ alta − 5;
  se è eco (o superiore), resta almeno eco finché p ≥ media − 5.
- Con qualcuno in casa la probabilità casa vale 100%; il blueprint usa allora
  l'attributo `away_probability` se l'integrazione lo fornisce, altrimenti considera
  100% quando esiste un'ETA (cioè qualcuno sta già tornando).
- Il calendario è valutato con lo stato attuale e `next_event` dell'helper
  `schedule`: se è spento e si accende prima dell'ETA, si assume che resti acceso
  all'arrivo (approssimazione adeguata per orizzonti di poche ore).

---

## Esempi multi‑zona e notte

Configurazione tipica, un'automazione per zona:

| Zona | Tipo | Calendario | Altro |
|---|---|---|---|
| Soggiorno + cucina | `day` | 07:00–23:00 | interruttore letto |
| Camera | `night` | 20:00–07:00 | interruttore letto |
| Bagno | `always` | 06:00–08:00, 19:00–23:30 | — |
| Studio | `day` | 08:30–18:00 feriali | sensore occupazione |

**Rientro previsto alle 23:30, casa vuota, pre‑riscaldo 30 min, p = 85%.**

- Alle 22:30 tutte le zone sono in `frost` (fuori dalla finestra di pre‑riscaldo).
- Alle 23:00 la **camera** passa a `comfort`: siamo nella finestra e il suo
  calendario (20–07) è attivo alle 23:30.
- Il **soggiorno** resta in `frost`: il suo calendario finisce alle 23:00, quindi
  all'arrivo non sarà in uso.
- Il **bagno** resta in `frost`: il suo calendario finisce alle 23:30, quindi
  all'ora di arrivo non è più attivo. Se il calendario arrivasse fino a mezzanotte,
  passerebbe a `comfort` alle 23:00 come la camera.

**Stessa sera, ma qualcuno è già in casa.** Camera in comfort (calendario attivo),
soggiorno in comfort fino alle 23:00 e poi nel preset "non in uso" (eco): il rientro
alle 23:30 non lo riporta in comfort perché la zona non è in calendario all'arrivo.

**Letto.** Quando accendi l'interruttore letto con qualcuno in casa: soggiorno e
studio passano al preset notte (eco), camera e bagno in comfort, indipendentemente
da calendari e rientri. Se tutti escono e il letto resta acceso per errore, viene
ignorato.

---

## Plugin termici: confronto e combinazione consigliata

Home Return Predictor sa **quando** arriva qualcuno; non sa **quanto ci mette una
stanza a scaldarsi**. Per i minuti di pre‑riscaldo si può usare un valore fisso o un
plugin termico.

| Progetto | Cosa fa | Compatibilità con VTherm a preset |
|---|---|---|
| [Ecronika/ha-preheat](https://github.com/Ecronika/ha-preheat) (v2.11.x) | Impara per zona massa termica, dispersione e tempo morto (profili Radiatore/Pavimento/AC). **Non scrive setpoint**: si controlla col suo blueprint (comfort/eco). Entità per zona `binary_sensor.<zona>_preheat_active`, `sensor.<zona>_next_preheat_start`, `sensor.<zona>_predicted_duration` (minuti); casa `sensor.preheat_house_next_arrival`, `binary_sensor.preheat_house_incoming`. Richiede un `binary_sensor` di occupazione per zona; `schedule_entity` e `global_presence` opzionali. Ha un proprio predittore di arrivo | **Buona** come fonte dei minuti: usa `sensor.<zona>_predicted_duration` nel nostro blueprint. **Non** usare contemporaneamente il suo blueprint di controllo (un solo controllore). La nostra ETA non può essere iniettata come obiettivo; `binary_sensor.house_someone_returning` può essere passato come indizio di presenza tramite il blueprint *presence bridge* (uso dedotto, non documentato da ha‑preheat) |
| [nic6911/ha-predictive-heating](https://github.com/nic6911/ha-predictive-heating) | Orientato al riscaldamento a pavimento; scrive i setpoint con `climate.set_temperature` (esiste una modalità advisory); entità `sensor.*_recommended_setpoint`, `*_predicted_temperature`; nessun ingresso di presenza/occupazione/calendario/arrivo | **Sconsigliato** con radiatori + VTherm: un cambio manuale di temperatura mette il preset VTherm a `none` e va in conflitto con il controllo a preset |
| [silkyclouds/ha-smart-preheat](https://github.com/silkyclouds/ha-smart-preheat) | Package YAML per una stanza; formula fisica `minuti = 60·ΔT·C / (P − UA·(T_target − T_est))` con C, UA, P tarati a mano; espone `sensor.preheat_minutes_needed`; scrive i setpoint da sé con `script.preheat_apply` | Solo come **fonte dei minuti**, rimuovendo lo script di controllo |

**Combinazione consigliata per radiatori + valvole termostatiche + VTherm**

1. **Home Return Predictor**: chi torna, quando, con che certezza.
2. **Blueprint VTherm zona**: unico controllore, solo preset.
3. **Minuti di pre‑riscaldo**: `sensor.<zona>_predicted_duration` di Ecronika
   ha‑preheat (appreso per zona, profilo radiatore) senza il suo blueprint di
   controllo; oppure, per iniziare, un numero fisso (30–45 min per radiatori).
4. **VTherm**: presenza e movimento **disattivati**; finestre e sicurezza **attive**.

---

## Esempi di automazioni

**Notifica quando qualcuno sta tornando (solo previsioni live)**

```yaml
alias: Notifica rientro previsto
triggers:
  - trigger: event
    event_type: home_return_predictor_return_detected
conditions:
  - condition: template
    value_template: "{{ not trigger.event.data.advisory }}"
actions:
  - action: notify.mobile_app_telefono
    data:
      title: In arrivo
      message: >
        {{ trigger.event.data.name }} dovrebbe arrivare alle
        {{ as_timestamp(trigger.event.data.eta) | timestamp_custom('%H:%M') }}
        ({{ (trigger.event.data.probability * 100) | round }}%).
```

**Usare il sensore in learning come semplice avviso**

```yaml
alias: Avviso rientro (anche in apprendimento)
triggers:
  - trigger: state
    entity_id: binary_sensor.giorgio_returning
    to: "on"
actions:
  - action: persistent_notification.create
    data:
      title: >
        {{ 'Previsione (in apprendimento)' if state_attr('binary_sensor.giorgio_returning', 'advisory') else 'Rientro previsto' }}
      message: >
        Probabilità {{ (state_attr('binary_sensor.giorgio_returning', 'probability') * 100) | round }}%,
        motivo: {{ state_attr('binary_sensor.giorgio_returning', 'reason') }}
```

---

## I primi giorni di apprendimento

Cosa guardare in `sensor.<p>_model_maturity` e negli altri sensori:

- **`episodes`**: numero di assenze usate. Dopo l'import dello storico dovrebbe già
  essere > 0; se è 0, controlla che la persona abbia storia nel recorder e che la
  zona di casa sia corretta.
- **`brier_score` vs `baseline_brier_score`**: il primo deve scendere sotto il
  secondo. Finché sono simili il modello non batte "la media" e resta in learning.
- **`brier_skill_score`**: > 0 significa che il modello aggiunge informazione; a 0.5
  il contributo della bravura alla maturità è pieno.
- **`sensor.<p>_return_reason`**: spiega la stima in forma compatta (in inglese), es.
  `left 06:10 (morning); approaching 45 km/h, 12.3 km; habit P 0.42, GPS LR 4.0`.
  È il modo più rapido per capire previsioni strane.
- Nel blueprint attiva temporaneamente **"Scrivi decisioni nel logbook"** per vedere
  perché ogni zona ha preso un certo preset.

Con abitudini regolari servono in genere 2–3 settimane per arrivare in live; con
turni variabili di più. Puoi forzare `live` dal select se ti fidi prima.

---

## Modello sfidante scikit-learn (opzionale)

Schema **champion/challenger**: il modello integrato (bayesiano a kernel, Python
puro) resta sempre il *campione*. Se abiliti l'opzione **Modello sfidante
scikit-learn** (disattivata di default), l'integrazione addestra in background uno
*sfidante* (`HistGradientBoostingClassifier`) sulle previsioni passate già
verificate, e lo usa **solo quando è dimostrabilmente migliore**.

- **Installazione a runtime**: all'attivazione Home Assistant installa
  `scikit-learn>=1.5` (circa 100 MB, rispettando i vincoli di versione di HA). Se
  l'installazione fallisce viene scritto un avviso nel log e si continua con il
  modello integrato; nulla si rompe.
- **Leggero anche su un N100**: un solo thread, modello piccolo (≤ 100 alberi da ≤ 15
  foglie), addestramento in background al massimo ogni 6 ore (o dopo molte nuove
  previsioni verificate), tipicamente ~1 s; un solo addestramento alla volta, mai
  nel loop di Home Assistant. Il modello non viene salvato: si riaddestra dal log
  persistito dopo il riavvio.
- **Cambio automatico solo se migliore**: lo sfidante diventa attivo solo se, su dati
  recenti **non visti in addestramento**, ha un Brier score migliore di almeno il
  **3 %** per **2 valutazioni consecutive** e con almeno **400 previsioni
  verificate**. Se poi peggiora, si torna **subito** al modello integrato.
- **Quale modello è attivo**: guarda gli attributi di
  `sensor.<p>_model_maturity`: `active_model` (`kernel_bayes` = integrato),
  `candidate_brier` (Brier di ogni candidato sull'ultima valutazione; più basso è
  meglio), `challenger_enabled`, `sklearn_available`. `active_model` compare anche
  su `binary_sensor.<p>_returning`, su `sensor.<p>_return_reason` (e nel testo
  della spiegazione) e nei diagnostics.

---

## Per sviluppatori: simulazione e test

Il cuore del modello (`custom_components/home_return_predictor/hrp_core/`) è Python
puro, senza import di Home Assistant.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements_test.txt
.venv/bin/python scripts/simulate.py          # mesi di dati sintetici, curva di accuratezza/maturità
.venv/bin/python -m pytest                    # tutti i test
.venv/bin/python -m pytest tests/model -q     # solo modello e preset_logic (senza HA)
.venv/bin/python -m pytest tests/blueprint -q # blueprint VTherm eseguito in un HA di test
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

- `tests/model/test_preset_logic.py`: tabella di verità della logica preset (zone
  giorno/notte/sempre, calendario valutato all'ETA, letto, priorità, mai declassare,
  isteresi senza ping‑pong, occupazione, protezioni).
- `tests/blueprint/`: carica il blueprint YAML, lo esegue in un Home Assistant di
  test con stati finti e verifica i preset richiesti, per garantire che il Jinja
  corrisponda a `preset_logic.py`.

---

## Limiti noti

- Serve un tracker affidabile per ogni persona; senza GPS il modello usa solo le
  abitudini.
- Il primo rientro dopo un cambio radicale di abitudini (nuovo lavoro, ferie) sarà
  previsto male; l'emivita regola quanto in fretta si adatta.
- La probabilità casa è "qualcuno rientra"; con più persone fuori l'ETA è quella del
  primo rientro previsto.
- Il calendario delle zone è letto con l'approssimazione descritta sopra (un blocco
  che inizia prima dell'ETA si assume ancora attivo all'arrivo).
- Le condizioni sulle entità opzionali (calendario, letto, occupazione) vengono
  rilevate subito sui cambi on/off; tutto il resto al più entro 2 minuti.
- Ospiti e persone non tracciate non sono visti dal modello: usa la fonte "qualcuno
  in casa" (es. un `input_boolean` o un gruppo) per includerli.
- Altri modelli: tutto il codice di Home Assistant parla con il modello tramite
  l'interfaccia astratta `ReturnModel` (`hrp_core/base.py`: `load_history`,
  `observe_transition`, `predict`, `log_prediction`, `resolve`, `maturity`,
  `to_dict`/`from_dict`); in futuro altri backend (es. modelli di sopravvivenza o
  altre librerie ML) possono essere aggiunti come nuove implementazioni o come
  ulteriori sfidanti, senza toccare entità e blueprint.
