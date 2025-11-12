# Deep Learning


# Database
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

