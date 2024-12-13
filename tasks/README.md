# Cvičení

## 1. cvičení: Blind search

Cílem prvního cvičení je implementace algoritmu pro slepé prohledávání v prostoru funkcí. Algoritmus systematicky
prozkoumává všechny body v definovaném prostoru, bez ohledu na jejich hodnotu, aby našel globální extrém. Tento přístup
je vyčerpávající, ale poskytuje přehled o celém prostoru funkcí. Výsledky budou vizualizovány, aby bylo možné vidět
pokrytí prostoru a nalezené řešení.

## 2. cvičení: Hill climbing

Cílem druhého cvičení je implementace algoritmu hill climbing pro prohledávání v prostoru funkcí. Algoritmus začíná
náhodným bodem a hledá v jeho okolí lepší řešení, dokud nenajde lokální extrém. Tento postup se opakuje, dokud nelze
nalézt lepší bod v bezprostředním okolí. Výsledky budou vizualizovány, aby byl patrný postup hledání.

## 3. cvičení: Simulované žíhání

Cílem třetího cvičení je implementace algoritmu simulovaného žíhání pro prohledávání v prostoru funkcí. Algoritmus
simuluje proces postupného ochlazování kovu a hledá globální optimum tak, že občas přijímá i horší řešení, čímž se vyhne
uvíznutí v lokálních extrémech. Postupně se snižuje „teplota“, což vede k jemnějšímu prohledávání okolí. Výsledky budou
vizualizovány, aby byl vidět průběh optimalizace.

## 4. cvičení: Genetické algoritmy

Cílem čtvrtého cvičení je implementace genetického algoritmu pro problém obchodního cestujícího. Na začátku je náhodně
vygenerováno několik měst, mezi kterými se hledá nejkratší cesta. Genetický algoritmus vytváří nové generace cest
křížením a mutacemi, aby postupně nalezl optimální trasu. Výsledky budou krásně vizualizovány v animaci.

## 5. cvičení: Diferenciální evoluce

Cílem pátého cvičení je implementace algoritmu diferenciální evoluce pro prohledávání v prostoru funkcí. Algoritmus
pracuje s populací jedinců, kteří se mezi sebou kříží a mutují, aby vytvářeli nové a lepší generace řešení. Postupně tak
populace konverguje k optimálnímu řešení v prostoru funkcí. Výsledky budou vizualizovány pro přehled o průběhu
evolučního procesu.

## 6. cvičení: Particle swarm optimization (PSO)

Cílem šestého cvičení je implementace algoritmu particle swarm optimization (PSO) pro prohledávání v prostoru funkcí.
Algoritmus simuluje chování hejna, kde každá částice představuje možné řešení a pohybuje se prostorem ovlivněná svým
osobním a globálním nejlepším nalezeným řešením. Částice tak postupně konvergují k optimálnímu bodu v prostoru. Výsledky
budou vizualizovány, aby bylo možné sledovat dynamiku hejna při hledání globálního extrému.

## 7. cvičení: Self oraganizing migration algorithm

Cílem sedmého cvičení je implementace algoritmu self organizing migration algorithm pro prohledávání v prostoru funkcí.
Jedn8 se o verzi "All in One", která v rámci jednoho kroku kombinuje migraci a organizaci částic. Tento algoritmus se
inspiroval chováním přirozených hejn a kolonií, kde jednotlivé částice kooperují s cílem najít optimální řešení. Na
začátku se každá částice umístí do náhodného bodu v prostoru funkcí, poté dochází k iterativnímu procesu, kde částice
přemísťují své pozice směrem k lepším sousedům nebo se organizují podle lokálních vůdců. Cílem je dosáhnout optimálního
rozložení částic v prostoru, aby se přiblížily globálnímu extrému. Výsledky budou vizualizovány, aby bylo možné
pozorovat dynamiku a efektivitu migrace a organizace částic.

## 8. cvičení: Ant colony hill climbing applied on TSP

Cílem osmého cvičení je implementace algoritmu ant colony hill climbing pro problém obchodního cestujícího. Algoritmus
kombinuje vlastnosti mravenčí kolonie a hill climbingu, kde mravenci hledají nejkratší cestu mezi městy a zároveň
využívají hill climbing k nalezení lokálních extrémů. Mravenci se pohybují mezi městy podle feromonové stopy, kterou
zanechávají, a postupně tak nalézají optimální trasu. Výsledky budou vizualizovány v animaci, aby bylo možné sledovat
postupné hledání nejkratší cesty.

## 9. cvičení: Firefly algorithm

Cílem devátého cvičení je implementace algoritmu firefly algorithm pro prohledávání v prostoru funkcí. Algoritmus
simuluje chování světlušek, které se přitahují k sobě na základě intenzity světla, které vysílají. Světlušky se pohybují
prostorem a snaží se nalézt globální extrém, který je reprezentován nejjasnější světluškou. Postupně se světlušky
přibližují k optimálnímu bodu v prostoru funkcí. Výsledky budou vizualizovány, aby bylo možné sledovat dynamiku
světlušek při hledání globálního extrému.

## 10. cvičení: Teaching learning based optimization

Cílem desátého cvičení je implementace algoritmu teaching learning based optimization pro prohledávání v prostoru
funkcí. Tentokrát se jedná o algoritmus inspirovaný procesem výuky a učení, kde studenti a učitelé spolupracují na
nalezení optimálního řešení. Studenti se pohybují v prostoru funkcí a snaží se nalézt globální extrém, zatímco učitelé
se snaží pomoci studentům k lepším řešením. Postupně se celá třída snaží konvergovat k optimálnímu bodu v prostoru
funkcí. Výsledky budou zapsány do jednoho Excel souboru (.xlsx), aby bylo možné analyzovat průběh optimalizace. Celkem
se provede daný počet experimentů, během kterého se ve více než trojrozměrném poli prostoru funkcí hledá optimální bod.
Pro každý algoritmus se vypočte průměrná hodnota a směrodatná odchylka z výsledků experimentů.