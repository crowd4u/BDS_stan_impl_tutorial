# BDS_stan_impl_tutorial

(このリポジトリは誠意更新中です)

Dawid-Skene法のベイズ近似推論（MCMC or 変分ベイズ）による実装を動かすためのDockerコンテナを提供します．

DS法の実装は [Paun et al. (2018)](https://aclanthology.org/Q18-1040/) のStan実装に基づきますが，Paunらの実装は事前分布の設定があまりよくない（と思う）ので，事前分布の設定方法の部分のみ田村が変更しています．

## 環境構築・起動方法

1. Dockerが使える環境を構築してください

2. 以下のコマンドでDockerを起動します
```sh
docker compose up -d
```
（初回は時間がかかります）

このコマンドを実行すると`http://localhost:8008`にJupyter Labが起動します．

3. 以下のコマンドでJupyter Labのtokenを確認します．
```sh
docker exec -it cmdstanpy_container jupyter lab list
```

4. `http://localhost:8008`にアクセスして，tokenを入力します．

5. `example.ipynb`に例があります．

## FAQ
## 新しいパッケージを追加するには
`Dockerfile`を編集した後，コンテナをリビルトしてください．
```sh
docker compose up -d --build
```

## 新しいモデルを開発する場合に注意することは？
新たに`Stan`のコードを書く場合（LLMに書かせる場合），利用しているStanのバージョンが古いことに注意してください．
Paun et al. (2018) に基づいて，`Stan 2.32.2`を利用しています．

以下のリファレンスを元にコーディングすると良いと思います．

https://mc-stan.org/docs/2_32/reference-manual/

