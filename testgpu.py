import tensorflow as tf

def verifica_gpu():
    gpus = tf.config.list_physical_devices('GPU')  # Restituisce la lista delle GPU disponibili <sup data-citation="1" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://stackoverflow.com/questions/38009682/how-to-tell-if-tensorflow-is-using-gpu-acceleration-from-inside-python-shell">1</a></sup><sup data-citation="2" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://stackoverflow.com/questions/51114771/how-to-ensure-tensorflow-is-using-the-gpu">2</a></sup><sup data-citation="5" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://it.eitca.org/intelligenza-artificiale/eitc-ai-tff-tensorflow-fondamenti/tensorflow-nel-laboratorio-di-google/come-sfruttare-gpus-e-tpus-per-il-tuo-progetto-ml/revisione-dell%27esame-su-come-sfruttare-gpus-e-tpus-per-il-tuo-progetto-ml/come-puoi-confermare-che-tensorflow-sta-accedendo-alla-gpu-in-google-colab/">5</a></sup>
    if gpus:
        print("La GPU è disponibile e riconosciuta da TensorFlow.")
        for gpu in gpus:
            print(f"Dispositivo rilevato: {gpu}")
    else:
        print("GPU non disponibile. Assicurati di aver installato la versione GPU di TensorFlow.")

if __name__ == "__main__":
    verifica_gpu()
