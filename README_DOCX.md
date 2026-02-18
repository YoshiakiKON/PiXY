Word (.docx) 生成手順

1) 必要条件
- Pandoc をインストール（https://pandoc.org/installing.html）
- Pandoc の `--citeproc` は Pandoc 2.11+ で利用可能（最近のPandoc推奨）

2) 使い方（PowerShell）
```powershell
cd C:\Python\PiXY
.\build_docx.ps1
```

オプション:
- SoftwareX の Word テンプレート（入手済みなら）を `build_docx.ps1` の `$referenceDoc` にパスとして設定してください。
- 画像は `documentation/images/` から参照されます。

注意:
- Windows環境でPandocがPATHにない場合は、インストーラ実行後に新しいPowerShellセッションを開いてください。
- 参考文献は `paper.bib` を使用して自動で組版されます。必要ならば `.bib` の修正を行ってください。
