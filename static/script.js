document.getElementById('predict-form').addEventListener('submit', async function(e){
  e.preventDefault();
  const form = e.currentTarget;
  const data = Object.fromEntries(new FormData(form).entries());

  // Coerce numeric inputs
  if(data.bmi) data.bmi = parseFloat(data.bmi);
  if(data.children) data.children = parseInt(data.children,10);

  const out = document.getElementById('output');
  out.textContent = 'Predicting...';

  try{
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(data)
    });

    if(!resp.ok){
      const err = await resp.json().catch(()=>({error:resp.statusText}));
      out.textContent = 'Error: ' + (err.error || JSON.stringify(err));
      return;
    }

    const json = await resp.json();
    out.textContent = JSON.stringify(json, null, 2);
  }catch(err){
    out.textContent = 'Request failed: ' + err.message;
  }
});
