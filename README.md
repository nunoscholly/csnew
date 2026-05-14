# STARTCrew

STARTCrew ist eine Streamlit-Web-App zur Verwaltung von Teams bei Events wie
Hackathons. Organisatoren können Events anlegen, Teilnehmende erfassen,
Skills bewerten, Teams zusammenstellen und sich über ein Machine-Learning-
Modell passende Kandidat:innen für jedes Team empfehlen lassen.

Entwickelt im Rahmen des Computer-Science-Projekts an der HSG St. Gallen.

---

## Live-Demo

Die App ist online verfügbar unter:
<https://wejwcj9v2357f7fsocxvhg.streamlit.app/>

Login mit dem Test-Account weiter unten ([Schritt 5](#lokale-ausführung)).

---

## Lokale Ausführung

1. Repository klonen und in das Projektverzeichnis wechseln.
2. Virtuelle Umgebung anlegen und Abhängigkeiten installieren:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Im Projekt-Root eine `.env`-Datei mit den folgenden Supabase-Zugangsdaten
   anlegen:

   ```env
   SUPABASE_URL=https://grdewbuenxzqtqerqbhv.supabase.co
   SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdyZGV3YnVlbnh6cXRxZXJxYmh2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzczODg5MDcsImV4cCI6MjA5Mjk2NDkwN30.nsJMDcuUfwEZSquG2ENhSm37pLVhKluwazboRL7qmPQ
   ```

   `SUPABASE_KEY` ist der öffentliche Anon-Key; der Zugriff wird über
   Row-Level-Security in Supabase abgesichert.

4. App starten:

   ```bash
   streamlit run app.py
   ```

5. Zum Einloggen mit bereits vorhandenen Seed-Daten kann folgender Test-Account
   verwendet werden:

   - **E-Mail:** `nunoscholly@gmail.com`
   - **Passwort:** `nuno1234`

---

## Hilfsmittelverzeichnis

Bei der Entwicklung dieser App wurden folgende KI-Hilfsmittel eingesetzt:

- **Claude (Anthropic):** Unterstützung beim Schreiben, Refactoring und
  Debuggen von Python-Code, bei der Gestaltung des ML-Recommenders, bei der Erstellung von Seed-Data sowie beim
  Verfassen dieser Dokumentation.
- **Weitere KI-Assistenten:** punktuell zur Recherche, für
  Code-Vorschläge und zur Fehlersuche.

Alle KI-generierten Inhalte wurden von den Projektmitgliedern geprüft,
angepasst und verantwortet.
