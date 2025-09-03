# Hair Color Demo (MediaPipe)

- Реалтайм с веб-камеры (HTTPS)
- Сегментация волос: MediaPipe **HairSegmenter**
- Режимы: сплошной цвет / омбре
- Интенсивность, размытие краёв (feather), простые H/S/L
- Снимок и сохранение PNG

## Запуск
Разверните как статический сайт (Netlify/Vercel/GitHub Pages) или локально через HTTPS:
```
npx http-server -S -C cert.pem -K key.pem
```
Откройте `index.html`, нажмите **Старт** и разрешите доступ к камере.
