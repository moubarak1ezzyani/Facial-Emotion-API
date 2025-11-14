# Deep Learning


## Database
### SQL >
Create a Database
```sql
CREATE DATABASE mon_projet_db;
``` 
Change password :

* Ouvrez : `C:\Program Files\PostgreSQL\...\data\pg_hba.conf`

    * Changez "METHOD" `host` : `scram-sha-256`/`md5` --> `trust`


    * `win + [R]` : `services.msc` => `restart`

    * Ouvrez un terminal : `cmd> C:\Program Files\PostgreSQL\...\bin`

```Bash
psql -U postgres
```
Une fois dans psql (vous verrez postgres=#), tapez cette commande SQL (remplacez nouveau_mot_de_passe par votre choix) :

```SQL
ALTER USER postgres PASSWORD 'nouveau_mot_de_passe_securise';
```
    Quittez psql : `\q`

Annuler les modifications :

    Retournez dans `pg_hba.conf` et remettez `scram-sha-256` / `md5` <-- `trust`.

    Redémarrez le service une dernière fois.

## Sensitive data  : (.env)
install package 
```bash
pip install python-dotenv
```


 ## create_async_engine vs create engine ?
***create_engine*** : 
* bloque votre application
* frameworks classiques (Flask, Django) 

***create_async_engine*** :
* obligatoirement avec des frameworks asynchrones : FastAPI,...etc
 
### async ?
**==> Gérer des milliers d'utilisateurs en même temps**
Pendant que l'Utilisateur A attend (`await`) le `db.commit()`, FastAPI utilise ce temps "mort" pour commencer à traiter la requête de l'Utilisateur B.

***ex :*** Ne pas utiliser `async` avec FastAPI, c'est un peu comme acheter une voiture de course et ne jamais dépasser les 30 km/h. 
