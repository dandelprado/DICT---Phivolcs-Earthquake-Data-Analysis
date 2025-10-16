(function(){
  const tabs = Array.from(document.querySelectorAll('.tab'));
  const panels = Array.from(document.querySelectorAll('.tab-panel'));

  function activate(tab){
    const targetId = tab.getAttribute('aria-controls');
    tabs.forEach(t=>{
      t.classList.toggle('active', t===tab);
      t.setAttribute('aria-selected', t===tab ? 'true' : 'false');
      t.tabIndex = t===tab ? 0 : -1;
    });
    panels.forEach(p=>{
      p.classList.toggle('active', p.id === targetId);
      p.hidden = (p.id !== targetId);
    });
    window.scrollTo({top:0, behavior:'instant'});
  }

  tabs.forEach(tab=>{
    tab.addEventListener('click', ()=> activate(tab));
    tab.addEventListener('keydown', (e)=>{
      const idx = tabs.indexOf(tab);
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        activate(tabs[(idx+1)%tabs.length]);
        tabs[(idx+1)%tabs.length].focus();
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        activate(tabs[(idx-1+tabs.length)%tabs.length]);
        tabs[(idx-1+tabs.length)%tabs.length].focus();
      }
      if (e.key === 'Home'){ e.preventDefault(); activate(tabs[0]); tabs[0].focus(); }
      if (e.key === 'End'){ e.preventDefault(); activate(tabs[tabs.length-1]); tabs[tabs.length-1].focus(); }
    });
  });

  // Initialize
  panels.forEach(p=> p.hidden = !p.classList.contains('active'));
})();
