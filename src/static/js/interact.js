document.addEventListener("DOMContentLoaded", () => {
    var socket = io.connect(document.URL);

    document.querySelectorAll("#location-form").forEach(e => {
        e.addEventListener("submit", ev => {
            ev.preventDefault();
            const formData = new FormData(e);
            data = {}; formData.forEach((val, key) => data[key] = val)

            socket.emit('start_calc', data);
            socket.on('path_update', function(data) {
                const segments = []
                if (data.start) segments.push(/*html*/`
                <p>Inizio elaborazione...</p>
                `)
                document.querySelector("#result").innerHTML = segments.join('')
            });
            socket.on('path_result', function(data) {
                const date = new Date(null);
                date.setSeconds(data.duration);
                document.querySelector("#result").innerHTML = `
                <p>Durata: ${date.toISOString().slice(11, 19)}</p>
                <p>Lunghezza: ${data.length / 1000}km</p>
                `
            });
            socket.on('interp_gdf', function(data) {
                document.querySelector('#result-table').innerHTML = data.table
            });
            socket.on('saved_csv', function(data) {
                document.querySelector("#result").appendChild(createElementFromHTML(/*html*/`
                <p>Salvato CSV in ${data.path}!</p>
                `));
            })
            socket.on('route_image', function(data) {
                const imageDiv = document.querySelector("#result-image")
                imageDiv.innerHTML = '';

                if (data.started) {
                    imageDiv.innerHTML = /*html*/`<div class="loading">Caricamento...</div>`;
                    return;
                }

                [ data.route_img, data.split_img, data.sample_img ].forEach(imgData => {
                    const blob = new Blob([imgData], {type:'image/png'});
                    const url = URL.createObjectURL(blob);
                    const img = new Image();
                    img.src = url;
                    imageDiv.appendChild(img);
                });
            })
        });
    });
});

function createElementFromHTML(htmlString) {
    var div = document.createElement('div');
    div.innerHTML = htmlString.trim();
  
    // Change this to div.childNodes to support multiple top-level nodes.
    return div.firstChild;
}